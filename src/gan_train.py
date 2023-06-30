import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch.distributed import is_initialized, get_rank

# from loaders.ultrasound_dataset import USZDataModule
# from transforms.ultrasound_transforms import ZGanTrainTransforms, ZGanEvalTransforms

from loaders.ultrasound_dataset import USDataModule, USDataModuleBlindSweep
from transforms.ultrasound_transforms import DiffusionEvalTransforms, DiffusionTrainTransforms

from callbacks.logger import GenerativeImageLoggerNeptune

from nets import generative

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy

from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

import pickle

def main(args):

    if(os.path.splitext(args.csv_train)[1] == ".csv"):
        df_train = pd.read_csv(args.csv_train)    
        df_val = pd.read_csv(args.csv_valid)    
        df_test = pd.read_csv(args.csv_test)
    else:
        df_train = pd.read_parquet(args.csv_train)
        df_val = pd.read_parquet(args.csv_valid)
        df_test = pd.read_parquet(args.csv_test)


    NN = getattr(generative, args.nn)
    model = NN(**vars(args))
    

    train_transform = DiffusionEvalTransforms()
    valid_transform = DiffusionEvalTransforms()

    usdata = USDataModule(df_train, df_val, df_test, batch_size=args.batch_size, mount_point=args.mount_point, num_workers=args.num_workers, img_column='img_path', drop_last=True, train_transform=train_transform, valid_transform=valid_transform, repeat_channel=False)

    

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)  
        image_logger = GenerativeImageLogger()  

    elif args.neptune_tags:
        logger = NeptuneLogger(
            project='ImageMindAnalytics/generative',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN']
        )
        image_logger = GenerativeImageLoggerNeptune(num_images=args.num_images)

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        max_steps=args.steps,
        callbacks=[early_stop_callback, checkpoint_callback, image_logger],
        accelerator='gpu', 
        devices=torch.cuda.device_count(),
        # plugins=[MixedPrecisionPlugin(precision='16-mixed', device='cuda')],
        strategy=DDPStrategy(find_unused_parameters=False)
    )
    
    trainer.fit(model, datamodule=usdata, ckpt_path=args.model)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Diffusion model training generating images 64x64')

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience steps for EarlyStopping', type=int, default=30)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)    
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=256)
    hparams_group.add_argument('--emb_dim', help='Latent dimension for generative network', type=int, default=128)
    hparams_group.add_argument('--in_channels', help='Input Channels ddpm', type=int, default=1)
    hparams_group.add_argument('--out_channels', help='Ouptut Channels ddpm', type=int, default=1)
    # hparams_group.add_argument('--perceptual_weight', help='Perceptual weight', type=float, default=0.001)
    # hparams_group.add_argument('--adversarial_weight', help='Adversarial weight', type=float, default=0.01)
    hparams_group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
    hparams_group.add_argument('--lr_d', help='Discriminator learning rate', type=float, default=1e-4)    
    hparams_group.add_argument('--num_train_timesteps', help='Num train steps for ddpml', type=int, default=1000)
    hparams_group.add_argument('--use_pre_trained', help='Use pre trained', type=int, default=0)
    

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="GanKL")
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')
    input_group.add_argument('--csv_test', required=True, type=str, help='Test CSV')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default=None)
    log_group.add_argument('--num_images', help='Max number of images', type=int, default=16)
    log_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    log_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="generative")


    args = parser.parse_args()

    main(args)
