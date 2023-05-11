import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from loaders.ultrasound_dataset import USDataModuleVolumes

from transforms.ultrasound_transforms import US3DTrainTransforms, US3DEvalTransforms
from nets.ga_net3d import GANet3D, GANet3DV0, GANet3DDot, GANet3DB3

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from sklearn.utils import class_weight

def main(args):

    train_fn = os.path.join(args.mount_point, 'CSV_files', args.csv_train)
    valid_fn = os.path.join(args.mount_point, 'CSV_files', args.csv_eval)
    test_fn = os.path.join(args.mount_point, 'CSV_files', 'c1_volumes_ga_boe_20220811_masked_resampled_128_spc075_notna_test.csv')    

    img_column='img_path'
    ga_column='ga_boe'
    id_column='StudyID'

    df_train = pd.read_csv(train_fn)
    df_val = pd.read_csv(valid_fn)      
    df_test = pd.read_csv(test_fn)
    
    usdata = USDataModuleVolumes(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column=img_column, ga_column=ga_column, id_column=id_column,
        train_transform=US3DTrainTransforms(), valid_transform=US3DEvalTransforms())


    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.out, args.nn),
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    
    if args.nn == "GANet3D":
        model = GANet3D(args)
    elif args.nn == "GANet3DV0":
        model = GANet3DV0(args)
    elif args.nn == "GANet3DDot":
        model = GANet3DDot(args)
    elif args.nn == "GANet3DB3":
        model = GANet3DB3(args)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)    

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=False)
    )
    trainer.fit(model, datamodule=usdata, ckpt_path=args.model)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='3D GANet Training')
    parser.add_argument('--nn', help='Type of neural network', type=str, default='GANet3D')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--model', help='Model path to continue training', type=str, default=None)
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=1000)    
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=24)
    parser.add_argument('--patience', help='Early stop patience', type=int, default=100)
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="ga_3d")

    parser.add_argument('--csv_train', help='CSV for training', type=str, default="c1_volumes_ga_boe_20220811_masked_resampled_128_spc075_notna_train_train.csv")    
    parser.add_argument('--csv_eval', help='CSV for evaluation', type=str, default="c1_volumes_ga_boe_20220811_masked_resampled_128_spc075_notna_train_eval.csv")    

    args = parser.parse_args()

    main(args)
