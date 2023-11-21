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

from utils import load_pkl
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


class MRI_Dataset(Dataset):
    def __init__(self, 
                 fpath,
                 corrupt_targets=True,
                 mode="train",
                 repetition_scans=6,
                 ) -> None:
        self.fpath = fpath
        self.corrupt_targets = corrupt_targets
        self.mode = mode
        img = []
        # import pdb
        # pdb.set_trace()
        if type(fpath) is list:
            for fp in fpath:
                im, sp = load_pkl(fp)
                # import pdb
                # pdb.set_trace()
                b, h, w = im.shape
                im = im[:,h%w:].reshape(b, h//w, w, w).transpose(1,0,2,3).reshape(b*(h//w), w, w)
                # idx = 0
                # fig, axes = plt.subplots(1,2);axes[0].imshow(im[idx],cmap="gray");axes[1].imshow(im[idx+5],cmap="gray");plt.show()
                img.append(im)
                self.spec.append(np.repeat(sp, h//w, axis=0))
            img = np.concatenate(img, axis=0)
            self.spec = np.concatenate(self.spec, axis=0)
        else:
            img, self.spec = load_pkl(fpath)

        # Convert to float32.
        assert img.dtype == np.uint8
        self.img = img.astype(np.float32) / 255.0 - 0.5
        self.all_indices = np.arange(self.img.shape[0])
        self.pairs = all_pairs(repetition_scans)

        # import pdb
        # pdb.set_trace()

    def __getitem__(self, idx):
        if self.mode == "train":
            img_idx = idx//len(self.pairs)*self.repetition_scans
            img = self.img[img_idx:img_idx+self.repetition_scans]
            spec = self.spec[img_idx:img_idx+self.repetition_scans]
            t = img[self.pairs[idx%len(self.pairs)][1]]
            img = img[self.pairs[idx%len(self.pairs)][0]]
            spec = spec[self.pairs[idx%len(self.pairs)][0]]
            img, trans = augment_data(img, spec)
            t = augment_data_same(t, trans=trans)
            inp, sv, sm = corrupt_data(img, spec, self.corrupt_params)

            h, w = inp.shape
            h = h//32*32-1
            w = w//32*32-1
            
            return inp[None, :h, :w], t[None, :h, :w], sv[None], sm[None]
        else:
            h, w = self.img[idx].shape
            h = h//32*32-1
            w = w//32*32-1
            return self.img[idx,:h,:w][None], self.img[idx,:h,:w][None], [], []
    
    def __len__(self):
        if self.mode == "train":
            return self.img.shape[0]//self.repetition_scans*len(self.pairs)
        else:
            return self.img.shape[0]


class OCT_Dataset(Dataset):
    def __init__(self, 
                 fpath,
                 augment_params=dict(),
                 corrupt_params=dict(),
                 corrupt_targets=True,
                 mode="train",
                 repetition_scans=6,
                 slice_direction="Ascan",
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
        # import pdb
        # pdb.set_trace()
        if type(fpath) is list:
            for fp in fpath:
                im = np.asanyarray(nib.load(fp).dataobj)
                if slice_direction == "enface":
                    im = im.transpose(1,0,2)
                img.append(im)
            img = np.concatenate(img, axis=0)
            self.sli_no = im.shape[0]
        else:
            img = np.asanyarray(nib.load(fpath).dataobj)
            self.sli_no = img.shape[0]

        self.img = img.astype(np.float32) / self.scale_factor - 0.5

    def __getitem__(self, idx):
        if self.mode == "train":
            img_idx = idx//(self.sli_no-1)*self.sli_no+idx%(self.sli_no-1)
            inp = self.img[img_idx]
            t = self.img[img_idx+1]
            h,w = inp.shape

            h = h//32*32-1
            w = w//32*32-1
            
            return inp[None, :h, :w], t[None, :h, :w]
        else:
            h, w = self.img[idx].shape
            h = h//32*32-1
            w = w//32*32-1
            return self.img[idx,:h,:w][None], self.img[idx,:h,:w][None]
    
    def __len__(self):
        if self.mode == "train":
            return int(len(self.img)/self.sli_no*(self.sli_no-1))
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

    dataset = OCT_Dataset(train_flist)
    testset = OCT_Dataset(test_flist, mode="test")
    dataloader = DataLoader(dataset, batch_size=args.bsz, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    model = AutoEncoder(1, 1)
    model = model.to(args.device)

    vxm_model = VxmDense(inshape=(dataset[0][0].shape[-2]+1, dataset[0][0].shape[-1]+1), 
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
            for step, (input_images, target_images) in enumerate(dataloader):
                input_images = input_images.float().to(args.device)
                target_images = target_images.float().to(args.device)

                if e>=args.wmp_epoch:
                    with torch.no_grad():
                        input_images_denoised = model(input_images)
                        target_images_denoised = model(target_images)

                    vxm_optimizer.zero_grad()
                    aligned_target_denoised, aligned_input_denoised, flow  = vxm_model(target_images_denoised.detach()+0.5, input_images_denoised.detach()+0.5)

                    vxm_loss = 0.5*vxm_losses[0](input_images_denoised+0.5, aligned_target_denoised)+\
                               0.5*vxm_losses[1](target_images_denoised+0.5, aligned_input_denoised)+\
                               args.vxm_smooth*vxm_losses[2](torch.zeros_like(flow), flow)

                    vxm_loss.backward()
                    vxm_optimizer.step()
                else:

                    vxm_optimizer.zero_grad()
                    aligned_target, aligned_input, flow  = vxm_model(target_images+0.5, input_images+0.5)

                    vxm_loss = 0.5*vxm_losses[0](input_images+0.5, aligned_target)+\
                               0.5*vxm_losses[1](target_images+0.5, aligned_input)+\
                               args.vxm_smooth*vxm_losses[2](torch.zeros_like(flow), flow)

                    vxm_loss.backward()
                    vxm_optimizer.step()

                with torch.no_grad():
                    pos_flow =  vxm_model.fullsize(vxm_model.integrate(flow)) if vxm_model.fullsize else vxm_model.integrate(flow)
                    aligned_target = vxm_model.transformer(target_images+0.5, pos_flow, mode="nearest")
                    neg_flow =  vxm_model.fullsize(vxm_model.integrate(-flow)) if vxm_model.fullsize else vxm_model.integrate(-flow)
                    aligned_input = vxm_model.transformer(input_images+0.5, neg_flow, mode="nearest")

                optimizer.zero_grad()
                # with torch.no_grad():
                pred_input = model(input_images)
                pred_target = model(target_images)

                # denoised = post_op(pred, spec_value=spec_val, spec_mask=spec_mask)
                loss = 0.5*criterion(pred_input, aligned_target[...,:-1,:-1].detach()-0.5)+\
                       0.5*criterion(pred_target, aligned_input[...,:-1,:-1].detach()-0.5)

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
                for step, (input_images, target_images) in enumerate(testloader):
                    input_images = input_images.float().to(args.device)
                    target_images = target_images.float().to(args.device)
    
                    with torch.no_grad():
                        pred = model(input_images)
    
                    pbar.update(1)
                    if step%6==0:
                        # import pdb
                        # pdb.set_trace()
                        prim = [input_images[0,0].detach().cpu().numpy(), 
                                pred[0,0].detach().cpu().numpy(), 
                                # denoised[0,0].detach().cpu().numpy(),
                                ]
                        spec = [fftshift2d(abs(np.fft.fft2(x))) for x in prim]
                        pimg = np.concatenate(prim, axis=1) + 0.5
                        simg = np.concatenate(spec, axis=1) * 0.03
                        img = np.concatenate([pimg, simg], axis=0)
                        plt.imsave(os.path.join(result_subdir, 'img{}.png'.format(step)), img, vmin=0.0, vmax=1.0, cmap="gray")
                    
                



if __name__=="__main__":
    main()
        
        
