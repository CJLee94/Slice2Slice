import os
from train import OCT_Dataset, fftshift2d
import argparse
from torch.utils.data import DataLoader
from networks import AutoEncoder
import json
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--test_data_fpath", nargs="+", help="the file of the data")
    parser.add_argument("--test_device", default="cuda", type=str, help="Device to train on")
    # parser.add_argument("--test_save_dir", default="./vxm_results")
    parser.add_argument("--test_ckpt", default=None, type=str)
    parser.add_argument("--config", default=None)
    parser.add_argument("--direction", default="Ascan")
    return parser.parse_args()

def main():
    args = parse_args()

    img_basename = os.path.basename(args.test_data_fpath[0])
    img_basename = os.path.splitext(img_basename)[0]

    testset = OCT_Dataset(args.test_data_fpath, mode="test", slice_direction=args.direction)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    args.__dict__["test_save_dir"] = os.path.split(args.test_ckpt)[0]

    # with open(os.path.join(args.test_save_dir, 'commandline_args.txt'), 'r') as f:
    #     args.__dict__.update(json.load(f))

    state_dict = torch.load(args.test_ckpt)

    model = AutoEncoder(1, 1)
    if "dn_model" in state_dict:
        model.load_state_dict(state_dict["dn_model"])
    else:
        model.load_state_dict(state_dict)
    model = model.to(args.test_device)
    model.eval()

    out = []

    with tqdm(total=len(testloader)) as pbar:
        pbar.set_description("Testing")
        for step, (input_images, target_images) in enumerate(testloader):
            input_images = input_images.float().to(args.test_device)
            target_images = target_images.float().to(args.test_device)

            with torch.no_grad():
                pred = model(input_images)

            out.append(np.clip(255.0*(pred[0,0].detach().cpu().numpy()+0.5), 0, 255))

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
                plt.imsave(os.path.join(args.test_save_dir, 'img{}.png'.format(step)), img, vmin=0.0, vmax=1.0, cmap="gray")
        out = np.stack(out)
        if args.direction=="enface":
            out = out.transpose(1,0,2)
        nib.save(nib.Nifti1Image(out.astype(np.uint8), np.eye(4)), os.path.join(args.test_save_dir, "Dn_"+img_basename+".nii"))



if __name__ == "__main__":
    main()