import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from nets import tsne_net
from loaders.array_dataset import TensorDatasetModule

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy

from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger


import pickle

def main(args):


    emb_data = pickle.load(open(args.emb_data, 'rb'))
    tsne_data = pickle.load(open(args.tsne_data, 'rb'))


    emb_data = np.array(emb_data)
    tsne_data = np.array(tsne_data)

    idx = np.random.permutation(len(emb_data))

    idx_train = int(len(emb_data)*.9)

    emb_data = emb_data[idx]
    tsne_data = tsne_data[idx]

    emb_data_train = torch.tensor(emb_data[0:idx_train])
    emb_data_valid = torch.tensor(emb_data[idx_train:])

    tsne_data_train = torch.tensor(tsne_data[0:idx_train])
    tsne_data_valid = torch.tensor(tsne_data[idx_train:])
    
    tensor_datamodule = TensorDatasetModule((emb_data_train, tsne_data_train), (emb_data_valid, tsne_data_valid))

    NN = getattr(tsne_net, args.nn)
    model = NN(**vars(args))

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    callbacks = [early_stop_callback, checkpoint_callback]

    logger = None
    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)  
        image_logger = DiffusionImageLogger()  
        callbacks.append(image_logger)

    elif args.neptune_tags:
        logger = NeptuneLogger(
            project='ImageMindAnalytics/TSNE',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN']
        )
        image_logger = DiffusionImageLoggerNeptune()
        callbacks.append(image_logger)

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        max_steps=args.steps,
        callbacks=callbacks,
        accelerator='gpu', 
        devices=torch.cuda.device_count(),
        # plugins=[MixedPrecisionPlugin(precision='16-mixed', device='cuda')],
        strategy=DDPStrategy(find_unused_parameters=False)
    )
    
    trainer.fit(model, datamodule=tensor_datamodule, ckpt_path=args.model)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Diffusion training')

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience steps for EarlyStopping', type=int, default=30)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)    
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=256)
    hparams_group.add_argument('--emb_dim', help='Embeding dimension', type=int, default=128)
    hparams_group.add_argument('--hidden_dim', help='Hidden dimension', type=int, default=128)
    # hparams_group.add_argument('--perceptual_weight', help='Perceptual weight', type=float, default=0.001)
    # hparams_group.add_argument('--adversarial_weight', help='Adversarial weight', type=float, default=0.01)
    hparams_group.add_argument('--momentum', help='Momentum for optimizer', type=float, default=0.9)
    hparams_group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
    hparams_group.add_argument('--loss', help='Type of loss function', choices=['kl', 'mse', 'kl+mse'], type=str, default="kl")
    hparams_group.add_argument('--kl_w', help='Weight for kl term in the loss function', type=float, default=1.0)    
    hparams_group.add_argument('--mse_w', help='Weight for mse term in the loss function', type=float, default=1.0)    
    # hparams_group.add_argument('--autoencoder_warm_up_n_epochs', help='Warmup epochs', type=float, default=10)        

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="TSNE_klmse")
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--emb_data', required=True, type=str, help='High dimensional data pickle file')
    input_group.add_argument('--tsne_data', required=True, type=str, help='TSNE features pickle file')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default=None)
    log_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    log_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="diffusion")


    args = parser.parse_args()

    main(args)
