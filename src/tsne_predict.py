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

from nets import tsne_net
from loaders.array_dataset import TensorDatasetModule

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import pickle
import SimpleITK as sitk

from tqdm import tqdm
import nrrd
import sys
def main(args):
    

    emb_data = pickle.load(open(args.emb_data, 'rb'))

    test_ds = torch.utils.data.TensorDataset(torch.tensor(np.array(emb_data)))
    test_loader = DataLoader(test_ds, batch_size=1024, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    NN = getattr(tsne_net, args.nn)
    model = NN.load_from_checkpoint(args.model)
    model.eval()
    model.cuda()

    with torch.no_grad():

        features = []
        for idx, X in tqdm(enumerate(test_loader), total=len(test_loader)):
            X = X[0]
            X = X.cuda(non_blocking=True)
            x = model(X)
            features.append(x.cpu())

    features = torch.cat(features, dim=0)
    features = features.numpy()

    out_dir = os.path.join(args.out, os.path.splitext(os.path.basename(args.model))[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    ext = os.path.splitext(args.emb_data)[1]
    out_name = os.path.join(out_dir, os.path.basename(args.emb_data)).replace(ext, ".pickle")
    pickle.dump(features, open(out_name, 'wb'))



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Diffusion predict')

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="TSNE_klmse")
    input_group.add_argument('--model', help='Model to predict', type=str, default= None)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)        
    input_group.add_argument('--emb_data', required=True, type=str, help='Test CSV')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")

    args = parser.parse_args()

    main(args)
