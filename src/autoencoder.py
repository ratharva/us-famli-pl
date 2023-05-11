import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from nets.effnet_decode import EffnetDecodeSimilarity, EffnetDecode, MonaiAutoEncoder
from loaders.ultrasound_dataset import USDataModule
from transforms.ultrasound_transforms import EffnetDecodeTrainTransforms, EffnetDecodeEvalTransforms, AutoEncoderTrainTransforms, AutoEncoderEvalTransforms
from callbacks.logger import EffnetDecodeImageLogger, AutoEncoderImageLogger




from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

# from azureml.core.run import Run
# run = Run.get_context()

def main(args):

    train_transform = None
    valid_transform = None
    image_logger = None

    if args.nn == "monai_autoencoder":
        train_transform = AutoEncoderTrainTransforms(256)
        valid_transform = AutoEncoderEvalTransforms(256)
        image_logger = AutoEncoderImageLogger(12)
        model = MonaiAutoEncoder(output_dim=args.output_dim, lr=args.lr, k=args.k)
    elif args.nn == "effnet_decode_similarity":
        model = EffnetDecodeSimilarity(args)    
    elif args.nn == "effnet_decode":
        image_logger = EffnetDecodeImageLogger()
        model = EffnetDecode(args, train_transform=EffnetDecodeTrainTransforms(256), valid_transform=EffnetDecodeEvalTransforms(256)        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )    
    

    if(os.path.splitext(args.csv_train)[1] == ".csv"):
        df_train = pd.read_csv(os.path.join(args.mount_point, args.csv_train))    
        df_val = pd.read_csv(os.path.join(args.mount_point, args.csv_valid))    
        df_test = pd.read_csv(os.path.join(args.mount_point, args.csv_test))
    else:
        df_train = pd.read_parquet(os.path.join(args.mount_point, args.csv_train))    
        df_val = pd.read_parquet(os.path.join(args.mount_point, args.csv_valid))    
        df_test = pd.read_parquet(os.path.join(args.mount_point, args.csv_test))

    usdata = USDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column='img_path', drop_last=True, train_transform=train_transform, valid_transform=valid_transform)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=30, verbose=True, mode="min")

    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)
    else:
        logger = NeptuneLogger(
                    project='juanprietob/AutoEncoder',
                    tags=args.tags,
                    api_key=os.environ['NEPTUNE_API_TOKEN']
                )
    

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, checkpoint_callback, image_logger],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=False)

    )
    trainer.fit(model, datamodule=usdata, ckpt_path=args.model)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Autoencoder Training')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    parser.add_argument('--model', help='Model to continue training', type=str, default= None)
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--tags', help='Tags for neptune', type=str, nargs="+", default=["autoencoder"])
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=256)
    parser.add_argument('--nn', help='Type of neural network', type=str, default="monai_autoencoder")
    parser.add_argument('--k', help='Top k values for effnet_decode approach. How many samples do you expect to be similar? k should be batch_size - similar samples in batch', type=int, default=128)
    parser.add_argument('--output_dim', help='Output dimension, used for the model effnet_simclr and effnet_decode', type=int, default=128)    
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="autoencoder")

    parser.add_argument('--pretrained_nn', help='Type of neural network', type=str, default=None)
    parser.add_argument('--pretrained_model', help='nn_model filename', type=str, default=None)

    parser.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    parser.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')
    parser.add_argument('--csv_test', required=True, type=str, help='Test CSV')

    args = parser.parse_args()

    main(args)
