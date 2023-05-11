import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch.distributed import is_initialized, get_rank
from torch.utils.data import Dataset, DataLoader

from loaders.ultrasound_dataset import USDataModule, USDataset
from transforms.ultrasound_transforms import DiffusionEvalTransforms, DiffusionTrainTransforms
# from callbacks.logger import DiffusionImageLogger

from nets import diffusion

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import pickle
import SimpleITK as sitk

from tqdm import tqdm
import nrrd
import sys
def main(args):

    if(os.path.splitext(args.csv)[1] == ".csv"):        
        df_test = pd.read_csv(os.path.join(args.mount_point, args.csv))
    else:        
        df_test = pd.read_parquet(os.path.join(args.mount_point, args.csv))


    NN = getattr(diffusion, args.nn)
    model = NN(**vars(args)).load_from_checkpoint(args.model)
    model.cuda()


    # train_transform = DiffusionTrainTransforms()
    valid_transform = DiffusionEvalTransforms()

    test_ds = USDataset(df_test, args.mount_point, img_column=args.img_column, transform=valid_transform, repeat_channel=False)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, persistent_workers=True, pin_memory=True, shuffle=False, prefetch_factor=args.prefetch_factor)

    with torch.no_grad():

        features = []
        for idx, X in tqdm(enumerate(test_loader), total=len(test_loader)):
            X = X.cuda(non_blocking=True)
            fname = df_test.loc[idx][args.img_column]
            if args.csv_root_path is not None:
                out_fname = fname.replace(args.csv_root_path, args.out)
            else:
                out_fname = os.path.join(args.out, fname)

            out_dir = os.path.dirname(out_fname)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            if not os.path.exists(out_fname):
                try:
                    X_hat, z_mu, z_sigma = model(X)
                    X_hat = X_hat[0].cpu().numpy()                    
                    header = nrrd.read_header(fname)
                    nrrd.write(out_fname, X_hat, header, index_order='C')
                except:
                    print("ERROR:", fname, file=sys.stderr)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Diffusion predict')

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="AutoEncoderKL")
    input_group.add_argument('--model', help='Model to predict', type=str, default= None)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)    
    input_group.add_argument('--prefetch_factor', help='Number of prefectch for loading', type=int, default=2)    
    input_group.add_argument('--csv', required=True, type=str, help='Test CSV')
    input_group.add_argument('--csv_root_path', type=str, default=None, help='Replaces a root path directory to empty, this is use to recreate a directory structure in the output directory, otherwise, the output name will be the name in the csv (only if csv flag is used)')

    input_group.add_argument('--img_column', type=str, default='img_path', help='Image column name in the csv')
    

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")

    args = parser.parse_args()

    main(args)
