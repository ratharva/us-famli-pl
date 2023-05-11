import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from ultrasound_dataset import USDataset, USEvalTransforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def main(args):

    train_fn = os.path.join(args.mount_point, 'CSV_files', 'imagenet_21k.parquet')

    img_column='img_path'
    ga_column='ga_boe'
    id_column='StudyID'

    df_train = pd.read_parquet(train_fn)

    train_ds = USDataset(df_train, transform=USEvalTransforms(224))
    train_dataloader = DataLoader(train_ds, batch_size=512, num_workers=32, persistent_workers=True, pin_memory=True)

    step = 0
    for batch in train_dataloader:
        step += 1
    


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='3D GANet Training')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--model', help='Model path to continue training', type=str, default=None)
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=2)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=4)
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="classification_efficientnet_b0")


    args = parser.parse_args()

    main(args)
