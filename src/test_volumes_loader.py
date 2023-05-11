import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from ultrasound_dataset import USDataModuleVolumes, US3DTrainTransforms, US3DEvalTransforms

from nets.ga_net3d import GANet3D

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from sklearn.utils import class_weight

def main(args):

    train_fn = os.path.join(args.mount_point, 'CSV_files', 'c1_volumes_ga_boe_20220811_masked_resampled_128_spc075_notna_train_train.csv')
    valid_fn = os.path.join(args.mount_point, 'CSV_files', 'c1_volumes_ga_boe_20220811_masked_resampled_128_spc075_notna_train_eval.csv')
    test_fn = os.path.join(args.mount_point, 'CSV_files', 'c1_volumes_ga_boe_20220811_masked_resampled_128_spc075_notna_test.csv')

    img_column='img_path'
    ga_column='ga_boe'
    id_column='StudyID'

    df_train = pd.read_csv(train_fn)
    df_val = pd.read_csv(valid_fn)      
    df_test = pd.read_csv(test_fn)
    
    usdata = USDataModuleVolumes(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column=img_column, ga_column=ga_column, id_column=id_column,
        train_transform=US3DTrainTransforms(), valid_transform=US3DEvalTransforms())

    usdata.setup()

    train_loader = usdata.train_dataloader()

    for batch in train_loader:
        v, ga = batch
        print(v.shape, ga)
    


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
