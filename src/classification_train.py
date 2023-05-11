import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from loaders.ultrasound_dataset import USDataModule
from transforms.ultrasound_transforms import USClassTrainTransforms, USClassEvalTransforms
from nets.classification import EfficientNet

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from sklearn.utils import class_weight

def main(args):
    
    # train_fn = os.path.join(args.mount_point, 'CSV_files', 'dataset_c_pairs_masked_resampled_256_spc075_uuids_study_uuid_train_train.csv')
    # valid_fn = os.path.join(args.mount_point, 'CSV_files', 'dataset_c_pairs_masked_resampled_256_spc075_uuids_study_uuid_train_eval.csv')
    # test_fn = os.path.join(args.mount_point, 'CSV_files', 'dataset_c_pairs_masked_resampled_256_spc075_uuids_study_uuid_test.csv')

    # train_fn = os.path.join(args.mount_point, 'CSV_files', 'C1_C2_Annotated_Frames_resampled_256_spc075_uuids_study_uuid_train.csv')
    # valid_fn = os.path.join(args.mount_point, 'CSV_files', 'C1_C2_Annotated_Frames_resampled_256_spc075_uuids_study_uuid_valid.csv')
    # test_fn = os.path.join(args.mount_point, 'CSV_files', 'C1_C2_Annotated_Frames_resampled_256_spc075_uuids_study_uuid_test.csv')
    # img_column='uuid_path'
    # class_column='class'

    if(os.path.splitext(args.csv_train)[1] == ".csv"):
        df_train = pd.read_csv(os.path.join(args.mount_point, args.csv_train))    
        df_val = pd.read_csv(os.path.join(args.mount_point, args.csv_valid))    
        df_test = pd.read_csv(os.path.join(args.mount_point, args.csv_test))
    else:
        df_train = pd.read_parquet(os.path.join(args.mount_point, args.csv_train))    
        df_val = pd.read_parquet(os.path.join(args.mount_point, args.csv_valid))    
        df_test = pd.read_parquet(os.path.join(args.mount_point, args.csv_test))

    img_column=args.img_column
    class_column=args.class_column

    unique_classes = np.sort(np.unique(df_train[class_column]))
    unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[class_column]))

    class_replace = {}
    for cn, cl in enumerate(unique_classes):
        class_replace[cl] = cn
    print(unique_classes, unique_class_weights, class_replace)    

    usdata = USDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column=img_column, class_column=class_column, train_transform=USClassTrainTransforms(256), valid_transform=USClassEvalTransforms(256))


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    model = EfficientNet(args, base_encoder=args.nn, num_classes=unique_classes.shape[0], class_weights=unique_class_weights)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=30, verbose=True, mode="min")

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


    parser = argparse.ArgumentParser(description='Autoencoder Training')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--model', help='Model path to continue training', type=str, default=None)
    parser.add_argument('--model_feat', help='Features model', type=str, default=None)
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=256)
    parser.add_argument('--nn', help='Type of neural network', type=str, default="efficientnet_b0")
    
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="classification_efficientnet_b0")
    parser.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    parser.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')
    parser.add_argument('--csv_test', required=True, type=str, help='Test CSV')
    parser.add_argument('--img_column', type=str, help='Name of column in CSV file', default='img_path')
    parser.add_argument('--class_column', type=str, help='Name of column in CSV file', default='class')



    args = parser.parse_args()

    main(args)
