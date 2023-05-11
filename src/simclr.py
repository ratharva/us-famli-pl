
import os
import argparse
import pandas as pd
from ultrasound_dataset import USDataModule

import torch
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.simclr.transforms import (
    SimCLREvalDataTransform, SimCLRTrainDataTransform)


def main(args):

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )
    
    ga = 'ga_boe'
    df_train = pd.read_csv(os.path.join(args.mount_point, 'CSV_files', 'extract_frames.csv'))
    df_train = df_train.query('197 <= ' + ga + ' and ' + ga + ' <= 300')

    df_val = pd.read_csv(os.path.join(args.mount_point, 'CSV_files', 'extract_frames_valid.csv'))
    df_val = df_val.query('197 <= ' + ga + ' and ' + ga + ' <= 300')
    
    df_test = pd.read_csv(os.path.join(args.mount_point, 'CSV_files', 'extract_frames_test.csv'))
    df_test = df_test.query('197 <= ' + ga + ' and ' + ga + ' <= 300')

    usdata = USDataModule(df_train, df_val, df_test, mount_point=args.mount_point, batch_size=args.batch_size, num_workers=args.num_workers)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=30, verbose=True, mode="min")
    
    usdata.train_transforms = SimCLRTrainDataTransform(224)
    usdata.val_transforms = SimCLREvalDataTransform(224)

    logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=False)
    )

    # model
    model = SimCLR(learning_rate=args.lr, hidden_mlp=args.hidden_mlp, temperature=args.temperature)

    # fit
    trainer = pl.Trainer()
    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Autoencoder Training')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    parser.add_argument('--model', help='Model to continue training', type=str, default= None)
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=256)
    
    parser.add_argument('--hidden_mlp', help='Dimension of the hidden MLP', type=int, default=2048)
    parser.add_argument('--temperature', help='Temperature, used for the model effnet_simclr', type=float, default=0.1)
    parser.add_argument('--feat_dim', help='Feature dimension', type=int, default=128)

    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="SimCLR")

    args = parser.parse_args()

    main(args)