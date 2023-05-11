import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch.utils.data import DataLoader

from loaders.ultrasound_dataset import USDatasetBlindSweep
from transforms.ultrasound_transforms import USEvalTransforms
import nets.ga_net as GA 

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies.ddp import DDPStrategy

from sklearn.utils import class_weight
from sklearn.metrics import classification_report

from tqdm import tqdm

import pickle

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    
    model = GA.GA_NetL()

    if args.model_pt:
        model.model.load_state_dict(torch.load(args.model_pt))
    elif args.model:
        model = model.load_from_checkpoint(args.model)

    model.cuda()
    model.eval()
    
    if(os.path.splitext(args.csv)[1] == ".csv"):        
        df_test = pd.read_csv(os.path.join(args.mount_point, args.csv))
    else:        
        df_test = pd.read_parquet(os.path.join(args.mount_point, args.csv))    

    test_ds = USDatasetBlindSweep(df_test, img_column=args.img_column, transform=USEvalTransforms())

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    with torch.no_grad():

        frame_predictions = []
        scores = []
        predictions = []        
        
        for idx, X in tqdm(enumerate(test_loader), total=len(test_loader)): 
            x, x_a, x_v, x_s, x_v_p, w_a = model(X.cuda(non_blocking=True).contiguous())

            frame_predictions.append(x_v_p)
            scores.append(x_s)
            predictions.append(x)
    
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)
        
    ext = os.path.splitext(args.csv)[1]

    df_test["pred"] = torch.cat(predictions, dim=0).cpu().numpy().reshape(-1)
    out_name = os.path.join(args.out, os.path.basename(args.csv)).replace(ext, ".pickle")
    pickle.dump((scores, frame_predictions), open(out_name, 'wb'))

    if(ext == ".csv"):        
        df_test.to_csv(os.path.join(args.out, os.path.basename(args.csv).replace('.csv', '_prediction.csv')), index=False)
    else:
        df_test.to_parquet(os.path.join(args.out, os.path.basename(args.csv).replace('.parquet', '_prediction.parquet')), index=False)
if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Classification predict')
    parser.add_argument('--csv', type=str, help='Parquet file for testing', required=True)    
    parser.add_argument('--img_column', type=str, help='Column name in the csv file with image path', default="uuid_path")        
    parser.add_argument('--model', help='Trained model', type=str, default=None)    
    parser.add_argument('--model_pt', help='Trained model pytorch format', type=str, default=None)    
    parser.add_argument('--out', help='Output directory', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=1)    

    args = parser.parse_args()

    main(args)