import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import matplotlib.pyplot as plt
import glob, os
import argparse
from tqdm import tqdm
import json
import nibabel as nib
from utils import load_pkl, calculate_psnr, calculate_ssim
from networks import AutoEncoder
from n2s.mask import Masker

os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
from vxm.torch.networks import VxmDense
from vxm.torch.losses import MSE as vxm_MSE
from vxm.torch.losses import NCC as vxm_NCC
from vxm.torch.losses import Grad as vxm_Grad
from vxm.torch.losses import Grad_v2 as vxm_Grad_v2


def rampup(epoch, rampup_length):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    return 1.0


def rampdown(epoch, num_epochs, rampdown_length):
    if epoch >= (num_epochs - rampdown_length):
        ep = (epoch - (num_epochs - rampdown_length)) * 0.5
        return math.exp(-(ep * ep) / rampdown_length)
    return 1.0


def fftshift2d(x, ifft=False):
    assert (len(x.shape) == 2) and all([(s % 2 == 1) for s in x.shape])
    s0 = (x.shape[0] // 2) + (0 if ifft else 1)
    s1 = (x.shape[1] // 2) + (0 if ifft else 1)
    x = np.concatenate([x[s0:, :], x[:s0, :]], axis=0)
    x = np.concatenate([x[:, s1:], x[:, :s1]], axis=1)
    return x


augment_translate_cache = dict()
def augment_data(img, spec):
    t = 0
    trans = np.random.randint(-t, t + 1, size=(2,))
    img = np.roll(img, trans, axis=(0, 1))
    #spec = spec * augment_translate_cache[key]
    return img, trans

def augment_data_same(img, trans):
    img = np.roll(img, trans, axis=(0, 1))
    return img

bernoulli_mask_cache = dict()
def corrupt_data(img, spec, params):
    keep = (np.random.uniform(0.0, 1.0, size=spec.shape)**2 < 0.5)
    smsk = keep.astype(np.float32)
    return img, spec, smsk


def all_pairs(n):
    couple = []

    for i in range(0, n):
        for j in range(0, n):
            if(i == j):
                continue
            couple.append([i, j])

    return couple


class OCT_Dataset(Dataset):
    def __init__(self, 
                 fpath,
                 augment_params=dict(),
                 corrupt_params=dict(),
                 corrupt_targets=True,
                 mode="train",
                 repetition_scans=6,
                 slice_direction=0,
                 scale_factor=255,
                 ) -> None:
        self.fpath = fpath
        self.augment_params = augment_params
        self.corrupt_params = corrupt_params
        self.corrupt_targets = corrupt_targets
        self.mode = mode
        self.repetition_scans = repetition_scans
        self.slice_direction=slice_direction
        self.scale_factor = scale_factor
        img = []
        if self.mode == "test":
            gt = []
        if type(fpath) is list:
            for fp in fpath:
                im = np.asanyarray(nib.load(fp).dataobj)
                if self.mode == "test":
                    gt_im = np.asanyarray(nib.load(fp.replace("noisy", "clean")).dataobj)
                    gt_im = gt_im/gt_im.max()*scale_factor
                if slice_direction == 1:
                    im = im.transpose(1,0,2)
                    if self.mode == "test":
                        gt_im.transpose(1,0,2)
                elif slice_direction == 2:
                    im = im.transpose(2,0,1)
                    if self.mode == "test":
                        gt_im.transpose(2,0,1)
                img.append(im)
                if self.mode == "test":
                    gt.append(gt_im)
            img = np.concatenate(img, axis=0)
            if self.mode == "test":
                gt = np.concatenate(gt, axis=0)
            self.sli_no = im.shape[0]
        else:
            img = np.asanyarray(nib.load(fpath).dataobj)
            self.sli_no = img.shape[0]

        self.img = img.astype(np.float32) / self.scale_factor - 0.5
        if self.mode == "test":
            self.gt = gt.astype(np.float32) / self.scale_factor - 0.5

    def __getitem__(self, idx):
        if self.mode == "train":
            img_idx = idx//(self.sli_no-2)*self.sli_no+idx%(self.sli_no-2)
            inp = self.img[img_idx]
            t = self.img[img_idx+1]
            t2 = self.img[img_idx+2]
            h,w = inp.shape

            h = h//32*32
            w = w//32*32
            
            return inp[None, :h, :w], t[None, :h, :w], t2[None, :h, :w]
        else:
            h, w = self.img[idx].shape
            h = h//32*32
            w = w//32*32
            return self.img[idx,:h,:w][None], self.gt[idx,:h,:w][None]
    
    def __len__(self):
        if self.mode == "train":
            return int(len(self.img)/self.sli_no*(self.sli_no-2))
        else:
            return len(self.img)
    

def fftshift3d(x, ifft):
    # assert len(x.shape) == 3
    s0 = (x.shape[-1] // 2) + (0 if ifft else 1)
    s1 = (x.shape[-2] // 2) + (0 if ifft else 1)
    x = torch.concat([x[..., s0:, :], x[..., :s0, :]], dim=-2)
    x = torch.concat([x[..., :, s1:], x[..., :, :s1]], dim=-1)
    return x


def post_op(denoised, spec_mask, spec_value):
    denoised_spec  = torch.fft.fft2(denoised)                            # Take FFT of denoiser output.
    denoised_spec  = fftshift3d(denoised_spec, False)                # Shift before applying mask.                                              # TF wants operands to have same type.
    # import pdb
    # pdb.set_trace()
    denoised_spec  = spec_value * spec_mask + denoised_spec * (1. - spec_mask)    # Force known frequencies.
    denoised       = torch.fft.ifft2(fftshift3d(denoised_spec, True)).float()  # Shift back and IFFT.
    return denoised
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_fpath", nargs="+", help="the file of the data")
    parser.add_argument("-t", "--test_fpath", nargs="+", default=None, help="the file of the data")
    parser.add_argument("--bsz", default=16, type=int, help="Training data batch size")
    parser.add_argument("--device", default="cuda", type=str, help="Device to train on")
    parser.add_argument("-e", "--no_epochs", default=300, type=int, help="number of epochs")
    parser.add_argument("--lr_max", default=1e-3, type=float, help="maximum of learning rate")
    parser.add_argument("--ckpt_period", default=10, type=int, help="checkpoint period")
    parser.add_argument("--save_dir", default="./vxm_results")
    parser.add_argument("--method", default="orig", type=str, help="training method: orig or neighbor")
    parser.add_argument("--int_down",default=1, type=int, help="integrate downsize for the vxm")
    parser.add_argument("--vxm_loss", default="NCC", type=str, help="loss function for the voxelmorph training")
    parser.add_argument("--wmp_epoch", default=1, type=int, help="the number of warm up epochs without denoising before voxelmorph")
    parser.add_argument("--direction", default=0, type=int, help="direction to sample the slice")
    # parser.add_argument("--grad", default="v1", type=str, help="Choose the version of the grad loss to control the smoothness of the voxelmorph displacement, can be either v1 or v2, v1 just caculate the gradient of the neighbors within 1 pixels while v2 calculate 50 pixels in dim 0 which takes the anisotropic feature of the OCT scan.")
    parser.add_argument("--vxm_smooth", default=1, type=float, help="the weight that control the smoothness of the voxelmorph displacement field")
    parser.add_argument("--smooth_order", default=1, type=int, help="the order of the smooth regularizer on the voxelmorph displacement field")
    return parser.parse_args()


def main():
    rampup_length = 10
    rampdown_length = 30
    adam_beta1_initial    = 0.9
    adam_beta1_rampdown   = 0.5
    adam_beta2            = 0.99

    args = parse_args()
    result_subdir = os.path.join(args.save_dir)
    os.makedirs(result_subdir, exist_ok=True)
    
    with open(os.path.join(result_subdir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    print("Data file: {}".format(args.data_fpath))
    # dataset = MRI_Dataset(args.data_fpath, corrupt_params=dict(type='bspec', p_at_edge=0.025), augment_params={'translate':64})
    # testset = MRI_Dataset(args.test_fpath, mode="test", corrupt_params=dict(type='bspec', p_at_edge=0.025), corrupt_targets=False)
    train_flist = []
    test_flist = []
    for df in args.data_fpath:
        if os.path.isdir(df):
            train_flist += glob.glob(os.path.join(df, "*"))
        else:
            train_flist.append(df)

    for df in args.test_fpath:
        if os.path.isdir(df):
            test_flist += glob.glob(os.path.join(df, "*"))
        else:
            test_flist.append(df)

    dataset = OCT_Dataset(train_flist, slice_direction=args.direction)
    testset = OCT_Dataset(test_flist, mode="test", slice_direction=args.direction)
    dataloader = DataLoader(dataset, batch_size=args.bsz, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    model = AutoEncoder(1, 1)
    model = model.to(args.device)

    vxm_model = VxmDense(inshape=(dataset[0][0].shape[-2], dataset[0][0].shape[-1]), 
                         nb_unet_features=[[16,32,32,32],[32,32,32,32,32,16,16]], 
                         sample_mode="bilinear", int_downsize=args.int_down, bidir=True)
    vxm_model = vxm_model.to(args.device)
    vxm_optimizer = torch.optim.Adam(vxm_model.parameters(), lr=1e-4)
    if args.vxm_loss == "MSE":
        vxm_losses = [vxm_MSE().loss, vxm_MSE().loss, vxm_Grad_v2('l2', loss_mult=args.int_down, smooth_order=args.smooth_order).loss]
    elif args.vxm_loss == "NCC":
        vxm_losses = [vxm_NCC().loss, vxm_NCC().loss, vxm_Grad_v2('l2', loss_mult=args.int_down, smooth_order=args.smooth_order).loss]
    
    # optimizer = Adam(model.parameters(), lr=2e-4, betas=[adam_beta1_initial, adam_beta2])
    optimizer = Adam(model.parameters(), lr=2e-4, betas=[0.9, 0.99])
    criterion = nn.MSELoss()
    masker = Masker()

    best_dn_loss = np.inf
    best_vxm_loss = np.inf

    for e in range(args.no_epochs):
        rampup_value = rampup(e, rampup_length)
        rampdown_value = rampdown(e, args.no_epochs, rampdown_length)
        adam_beta1 = (rampdown_value * adam_beta1_initial) + ((1.0 - rampdown_value) * adam_beta1_rampdown)
        learning_rate = rampup_value * rampdown_value * args.lr_max
        for g in optimizer.param_groups:
            g['lr'] = learning_rate
            g['betas'] = [adam_beta1, 0.99]
        print("Learning rate is set to be {}".format(learning_rate))
        train_loss = 0
        train_vxm_loss = 0
        model.train()
        vxm_model.train()
        with tqdm(total=len(dataloader)) as pbar:
            pbar.set_description("Epoch {}".format(e))
            for step, (input_images, target_images, target2_images) in enumerate(dataloader):
                input_images = input_images.float().to(args.device)
                target_images = target_images.float().to(args.device)
                target2_images = target2_images.float().to(args.device)

        
                with torch.no_grad():
                    input_images_denoised = model(input_images)
                    target_images_denoised = model(target_images)
                    target2_images_denoised = model(target2_images)

                vxm_optimizer.zero_grad()
                aligned_input_denoised, aligned_target_denoised, flow  = vxm_model(input_images_denoised.detach()+0.5, target_images_denoised.detach()+0.5)
                aligned_target2_denoised, aligned_target_denoised2, flow2  = vxm_model(target2_images_denoised.detach()+0.5, target_images_denoised.detach()+0.5)

                vxm_loss = 0.25*vxm_losses[0](input_images_denoised+0.5, aligned_target_denoised)+\
                            0.25*vxm_losses[1](target_images_denoised+0.5, aligned_input_denoised)+\
                            0.25*vxm_losses[0](target_images_denoised+0.5, aligned_target2_denoised)+\
                            0.25*vxm_losses[1](target2_images_denoised+0.5, aligned_target_denoised2)+\
                            0.5*args.vxm_smooth*vxm_losses[2](torch.zeros_like(flow), flow)+\
                            0.5*args.vxm_smooth*vxm_losses[2](torch.zeros_like(flow2), flow2)

                vxm_loss.backward()
                vxm_optimizer.step()

                with torch.no_grad():
                    pos_flow =  vxm_model.fullsize(vxm_model.integrate(flow)) if vxm_model.fullsize else vxm_model.integrate(flow)
                    aligned_input = vxm_model.transformer(input_images+0.5, pos_flow, mode="nearest")-0.5
                    pos_flow2 =  vxm_model.fullsize(vxm_model.integrate(flow2)) if vxm_model.fullsize else vxm_model.integrate(flow2)
                    aligned_target2 = vxm_model.transformer(target2_images+0.5, pos_flow2, mode="nearest")-0.5

                optimizer.zero_grad()
                # with torch.no_grad():
                pred_input = model(target_images)
                target = (target_images+aligned_input.detach()+aligned_target2.detach())/3.0

                # denoised = post_op(pred, spec_value=spec_val, spec_mask=spec_mask)
                loss = criterion(pred_input, target.detach())
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_vxm_loss += vxm_loss.item()
                pbar.set_postfix({"avg_vxm_loss":train_vxm_loss/(step+1), "avg_loss":train_loss/(step+1)})

                pbar.update(1)

        test_db_clamped = 0.0
        if e%args.ckpt_period == args.ckpt_period-1:
            model.eval()
            vxm_model.eval()
            torch.save({"dn_model": model.state_dict(), "vxm_model":vxm_model.state_dict()}, os.path.join(result_subdir, 'ckpt_{}.pth'.format(e)))
            if train_loss/(step+1) < best_dn_loss:
                best_dn_loss = train_loss/(step+1)
                torch.save({"dn_model": model.state_dict(), "vxm_model":vxm_model.state_dict()}, os.path.join(result_subdir, 'best_{}.pth'.format(e)))
            with tqdm(total=len(testloader)) as pbar:
                pbar.set_description("Testing")
                running_psnr = 0
                running_ssim = 0
                for step, (input_images, target_images) in enumerate(testloader):
                    input_images = input_images.float().to(args.device)
                    target_images = target_images.float().to(args.device)
    
                    with torch.no_grad():
                        pred = model(input_images)

                    pred255 = np.clip(255*(pred.detach().cpu().numpy().transpose(0,2,3,1)+0.5), 0, 255).astype(np.uint8)
                    target255 = np.clip(255*(target_images.detach().cpu().numpy().transpose(0,2,3,1)+0.5), 0, 255).astype(np.uint8)
                    running_psnr += calculate_psnr(target255.squeeze(), pred255.squeeze())
                    running_ssim += calculate_ssim(target255.squeeze(), pred255.squeeze())
                    # SSIM.update(pred, target_images)

                    pbar.set_postfix({"PSNR":running_psnr/(step+1), "SSIM":running_ssim/(step+1)})
    
                    pbar.update(1)
                    if step%50==0:
                        # import pdb
                        # pdb.set_trace()
                        prim = [input_images[0,0,:-1,:-1].detach().cpu().numpy(), 
                                pred[0,0,:-1,:-1].detach().cpu().numpy(), 
                                # denoised[0,0].detach().cpu().numpy(),
                                ]
                        spec = [fftshift2d(abs(np.fft.fft2(x))) for x in prim]
                        pimg = np.concatenate(prim, axis=1) + 0.5
                        simg = np.concatenate(spec, axis=1) * 0.03
                        img = np.concatenate([pimg, simg], axis=0)
                        plt.imsave(os.path.join(result_subdir, 'img{}.png'.format(step)), img, vmin=0.0, vmax=1.0, cmap="gray")
                    
                



if __name__=="__main__":
    main()
        
        
