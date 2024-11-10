# save dpb weight
python -u -m torch.distributed.launch --nproc_per_node 1 save_rpb_weight.py --cfg configs/crossformer/tiny_patch4_group7_224.yaml \
--batch-size 128 --data-path /home1/yanweicai/DATA/tta/clip_based_adaptation/imagenet --eval --resume ./model_ckpt/crossformer-t.pth --use_dpb

# inference with dpb weight
python -u -m torch.distributed.launch --nproc_per_node 1 main.py --cfg configs/crossformer/tiny_patch4_group7_224.yaml \
--batch-size 128 --data-path /home1/yanweicai/DATA/tta/clip_based_adaptation/imagenet --eval --resume ./model_ckpt/crossformer-t.pth --use_dpb

# inference with rpb weight
python -u -m torch.distributed.launch --nproc_per_node 1 main.py --cfg configs/crossformer/tiny_patch4_group7_224.yaml \
--batch-size 128 --data-path /home1/yanweicai/DATA/tta/clip_based_adaptation/imagenet --eval --resume ./model_ckpt/cros_tiny_patch4_group7_224_rpb.pth 