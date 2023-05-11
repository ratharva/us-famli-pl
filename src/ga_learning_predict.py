import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from loaders.ultrasound_dataset import USDatasetVolumes
from transforms.ultrasound_transforms import US3DEvalTransforms

from nets.ga_net3d import GANet3D, GANet3DV0, GANet3DDot, GANet3DB3

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from sklearn.utils import class_weight
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm

def main(args):

    if args.nn == "GANet3D":
        model = GANet3D(args)
    elif args.nn == "GANet3DV0":
        model = GANet3DV0(args)
    elif args.nn == "GANet3DDot":
        model = GANet3DDot(args)
    elif args.nn == "GANet3DB3":
        model = GANet3DB3(args)

    model = model.load_from_checkpoint(args.model)
    model.eval()
    model.cuda() 
    
    df_test = pd.read_csv(args.csv)

    test_ds = USDatasetVolumes(df_test, mount_point=args.mount_point, img_column=args.img_column, ga_column=args.ga_column, id_column=args.id_column, max_seq=-1, transform=US3DEvalTransforms(128))  

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4, persistant_workers=True)

    with torch.no_grad():

        ga_pred = []
        feat_att = []
        scores = []
        feat = []
        feat_pred = []

        pbar = tqdm(enumerate(test_loader), total=len(test_loader))

        for idx, batch in pbar:
            X, Y = batch
            X = X.cuda(non_blocking=True).contiguous()           
            x, x_a, x_s, x_v, x_v_p = model(X)
            x = x.cpu()

            ga_pred.append(x)
            feat_att.append(x_a.cpu())
            scores.append(x_s.cpu().numpy())
            feat.append(x_v.cpu().numpy())
            feat_pred.append(x_v_p.cpu().numpy())

            pbar.set_description("pred: %f, ga: %f, error: %f" % (x, Y, x-Y))

        ga_pred = torch.cat(ga_pred, dim=0)
        feat_att = torch.cat(feat_att, dim=0)

        ga_pred = ga_pred.numpy()
        feat_att = feat_att.numpy()        

    out_dir = os.path.join(args.out, os.path.splitext(os.path.basename(args.model))[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_name = os.path.join(out_dir, os.path.basename(args.csv)).replace(".csv", ".pickle")

    pickle.dump((ga_pred, feat_att, scores, feat, feat_pred), open(out_name, 'wb'))


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Autoencoder Training')    
    parser.add_argument('--csv', type=str, help='CSV file for testing', required=True)
    parser.add_argument('--nn', help='Type of neural network', type=str, default='GANet3D')
    parser.add_argument('--img_column', type=str, help='CSV file for testing', default="img_path")
    parser.add_argument('--ga_column', type=str, help='GA column prediction', default="ga_boe")    
    parser.add_argument('--id_column', type=str, help='Study id column for grouping', default="StudyID")
    parser.add_argument('--model', help='Trained model checkpoint', type=str, required=True)
    parser.add_argument('--out', help='Output directory, the model name will be appended as a directory', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=1)


    args = parser.parse_args()

    main(args)
