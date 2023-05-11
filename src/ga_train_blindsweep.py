import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch.utils.data import DataLoader

from loaders.ultrasound_dataset import USDataModuleBlindSweep
from transforms.ultrasound_transforms import USTrainGATransforms, USEvalGATransforms
import nets.ga_net as GA 

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy

from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from sklearn.utils import class_weight
from sklearn.metrics import classification_report

from tqdm import tqdm

import pickle

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from callbacks.logger import BlindSweepImageLogger

def main(args):


    if(os.path.splitext(args.csv_train)[1] == ".csv"):
        df_train = pd.read_csv(os.path.join(args.mount_point, args.csv_train))    
        df_val = pd.read_csv(os.path.join(args.mount_point, args.csv_valid))    
        df_test = pd.read_csv(os.path.join(args.mount_point, args.csv_test))
    else:
        df_train = pd.read_parquet(os.path.join(args.mount_point, args.csv_train))    
        df_val = pd.read_parquet(os.path.join(args.mount_point, args.csv_valid))    
        df_test = pd.read_parquet(os.path.join(args.mount_point, args.csv_test))

    
    model = GA.GA_NetL()

    if args.model_pt:
        model.model.load_state_dict(torch.load(args.model_pt))


    train_transform = USTrainGATransforms(num_frames=args.num_frames)
    valid_transform = USEvalGATransforms(num_frames=args.num_frames)

    usdata = USDataModuleBlindSweep(df_train, df_val, df_test, mount_point=args.mount_point, batch_size=args.batch_size, num_workers=args.num_workers, 
        img_column='img_path', id_column='PID_date', ga_column='ga_edd', drop_last=True, max_sweeps=2,
        train_transform=train_transform, valid_transform=valid_transform)

    

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)  
        image_logger = BlindSweepImageLogger()  

    elif args.neptune_tags:
        logger = NeptuneLogger(
            project='ImageMindAnalytics/GA',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN']
        )
        image_logger = BlindSweepImageLogger()

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        max_steps=args.steps,
        callbacks=[early_stop_callback, checkpoint_callback],
        accelerator='gpu', 
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=False)
    )
    
    trainer.fit(model, datamodule=usdata, ckpt_path=args.model)



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='GA training')

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience steps for EarlyStopping', type=int, default=30)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)    
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=256)    
    hparams_group.add_argument('--num_frames', help='Number of frames per sweep', type=int, default=50)    

    input_group = parser.add_argument_group('Input')
    # input_group.add_argument('--nn', help='Type of neural network', type=str, default="GA_NetL")
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    input_group.add_argument('--model_pt', help='Model to initialize weights', type=str, default= None)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')
    input_group.add_argument('--csv_test', required=True, type=str, help='Test CSV')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default=None)
    log_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    log_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="ga")


    args = parser.parse_args()

    main(args)