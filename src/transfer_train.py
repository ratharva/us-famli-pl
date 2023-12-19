import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from loaders.ultrasound_dataset_classification import USDataModule
from transforms.ultrasound_transforms import USClassTrainTransforms, USClassEvalTransforms
from nets import classification_old

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from sklearn.utils import class_weight
import transferModel
torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    
    if(os.path.splitext(args.csv_train)[1] == ".csv"):
        df_train = pd.read_csv(os.path.join(args.mount_point, args.csv_train))    
        df_val = pd.read_csv(os.path.join(args.mount_point, args.csv_valid))    
        df_test = pd.read_csv(os.path.join(args.mount_point, args.csv_test))
    else:
        df_train = pd.read_parquet(os.path.join(args.mount_point, args.csv_train))    
        df_val = pd.read_parquet(os.path.join(args.mount_point, args.csv_valid))    
        df_test = pd.read_parquet(os.path.join(args.mount_point, args.csv_test))

    img_column=args.img_column
    # class_column=args.class_column

    # unique_classes = np.sort(np.unique(df_train[class_column]))
    # unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[class_column]))

    # class_replace = {}
    # for cn, cl in enumerate(unique_classes):
    #     class_replace[cl] = cn
    # print(unique_classes, unique_class_weights, class_replace)  

    # myLabels = ['No structures visible', 'Head Visible',
    #    'Abdomen Visible', 'Chest Visible', 'Femur Visible',
    #    'Other arm or leg bones visible', 'Umbilical cord visible',
    #    'Amniotic fluid visible', 'Placenta visible', 'Fetus or CRL visible',
    #    'Maternal bladder visible', 'Head Measurable', 'Abdomen Measurable',
    #    'Femur Measurable']  

    # myLabels = ['No structures visible', 'Head Visible',
    #    'Abdomen Visible', 'Amniotic fluid visible', 
    #    'Placenta visible']  
    myLabels = ['No structures visible', 'Head Visible',
       'Abdomen Visible', 'Femur Visible', 
       'Placenta visible']  

    usdata = USDataModule(df_train, df_val, df_test, label_columns=myLabels, batch_size=args.batch_size, num_workers=args.num_workers, img_column=img_column, train_transform=USClassTrainTransforms(256), valid_transform=USClassEvalTransforms(256), test_transform=USClassEvalTransforms(256))


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    # NN = getattr(transferModel.EfficientNetTransfer, args.nn)

    # args_params = vars(args)
    # # args_params['class_weights'] = unique_class_weights
    # # args_params['num_classes'] = len(unique_class_weights)

    # model = NN(**args_params)
    # # model = transferModel.EfficientNetTransfer()

    model = transferModel.EfficientNetTransfer(base_encoder=args.base_encoder, num_classes=len(myLabels), ckpt_path=args.ckpt_path)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=30, verbose=True, mode="min")


    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)    
    elif args.neptune_tags:
        logger = NeptuneLogger(
            project='ImageMindAnalytics/Classification',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN']
        )        

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=False)
    )
    trainer.fit(model, datamodule=usdata, ckpt_path=args.model)
    # trainer.test(datamodule=usdata)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Clasification Training')


    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--model', help='Model path to continue training', type=str, default=None)
    input_group.add_argument('--model_feat', help='Features model', type=str, default=None)
    input_group.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')
    input_group.add_argument('--csv_test', required=True, type=str, help='Test CSV')
    input_group.add_argument('--img_column', type=str, help='Name of column in CSV file', default='img_path')
    # input_group.add_argument('--class_column', type=str, help='Name of column in CSV file', default='class')
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")

    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default=None)
    log_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    log_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="classification_efficientnet_b0")

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    hparams_group.add_argument('--weight_decay', help='Weight_decay for optimizer', type=float, default=0.01)    
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=256)
    hparams_group.add_argument('--nn', help='Type of PL neural network', type=str, default="EfficientNet")
    hparams_group.add_argument('--base_encoder', help='Type of torchvision neural network', type=str, default="efficientnet_b0")
    hparams_group.add_argument('--ckpt_path', help='path to pre_trained model', type=str, default="./cluster_annotation_results_20230424.csv")
    
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    

    

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output', type=str, default="./")
    
    


    args = parser.parse_args()

    main(args)
