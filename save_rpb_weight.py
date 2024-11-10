import os
import time
import warnings
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter, NativeScaler

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_checkpoint_only, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

warnings.filterwarnings("ignore", module="PIL")

def parse_option():
    parser = argparse.ArgumentParser('CrossFormer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-set', type=str, default='imagenet', help='dataset to use')
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='native', choices=['native', 'O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', default='debug', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--num_workers', type=int, default=8, help="")
    parser.add_argument('--warmup_epochs', type=int, default=20, help="#epoches for warm up")
    parser.add_argument('--epochs', type=int, default=300, help="#epoches")
    parser.add_argument('--lr', type=float, default=5e-4, help="max learning rate for training")
    parser.add_argument('--min_lr', type=float, default=5e-6, help="min learning rate for training")
    parser.add_argument('--warmup_lr', type=float, default=5e-7, help="learning rate to start warmup")
    parser.add_argument('--weight_decay', type=float, default=5e-2, help="l2 reguralization")

    # local rank is obtained using os.environ in newr version
    # parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    parser.add_argument("--img_size", type=int, default=224, help='input resolution for image')
    parser.add_argument("--embed_dim", type=int, nargs='+', default=None, help='size of embedding')
    parser.add_argument("--impl_type", type=str, default='', help='options to use for different methods')

    # arguments relevant to our experiment
    parser.add_argument('--group_type', type=str, default='constant', help='group size type')
    parser.add_argument('--use_cpe', action='store_true', help='whether to use conditional positional encodings')
    parser.add_argument('--pad_type', type=int, default=0, help='0 to pad in one direction, otherwise 1')
    parser.add_argument('--no_mask', action='store_true', help='whether to use mask after padding')
    parser.add_argument('--adaptive_interval', action='store_true', help='interval change with the group size')
    parser.add_argument('--use_dpb', action='store_true', help='whether to use dynamic positional bias')
    
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(args, config):
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, args)
    model.cuda()
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    

    if config.MODEL.RESUME:
        # load model
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.info(msg)
        del checkpoint
        torch.cuda.empty_cache()
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}'")

        images = torch.randn(1, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE).cuda()
        images = images.cuda(non_blocking=True)

        # compute output
        # with torch.cuda.amp.autocast(enabled=(config.AMP_OPT_LEVEL=="native")):
        model(images)
        
        to_save = {'model': model.state_dict()}
        torch.save(to_save, os.path.join('model_ckpt', f'{config.MODEL.NAME}_rpb.pth'))

        return




if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0" and config.AMP_OPT_LEVEL != "native":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    config.defrost()

    config.freeze()

    os.makedirs(config.LOG_OUTPUT,    exist_ok=True)
    os.makedirs(config.WEIGHT_OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.LOG_OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.LOG_OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(args, config)
