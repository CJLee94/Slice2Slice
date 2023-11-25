# Self-supervised OCT Image Denoising with Slice-to-Slice Registration and Reconstruction

Required Package:
1. pytorch
2. cv2
3. tqdm
4. nibabel

To train the proposed model

`python train_ixi.py -d [directory that contains the training images] -t [directory that contains the testing images] --lr_max 0.0002 --ckpt_period 1 --save_dir ./results/slice2slice_default --vxm_smooth 1 --smooth_order 1 --wmp_epoch 0 --bsz 16 -e 300 --direction 0`
