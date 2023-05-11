import os

import argparse
import numpy as np 
import pandas as pd
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from nets.effnet_decode import EffnetDecodeSimilarity, EffnetDecode, MonaiAutoEncoder
from loaders.ultrasound_dataset import USDataset
from transforms.ultrasound_transforms import EffnetDecodeTestTransforms, AutoEncoderTestTransforms

from tqdm import tqdm

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):


    test_transform = None
    if args.nn == "monai_autoencoder":
        model = MonaiAutoEncoder().load_from_checkpoint(args.model)
        test_transform = AutoEncoderTestTransforms(256)
    elif args.nn == "effnet_decode_similarity":
        model = EffnetDecodeSimilarity(args).load_from_checkpoint(args.model)
    elif args.nn == "effnet_simclr":
        model = EffnetSimCLR(args).load_from_checkpoint(args.model)    
    elif args.nn == "effnet_decode":
        model = EffnetDecode(args).load_from_checkpoint(args.model, args=args)
        test_transform = EffnetDecodeTestTransforms(256)
    
    model.eval()
    model.cuda()
    print(model)
    
    if(os.path.splitext(args.csv)[1] == ".csv"):
        df_test = pd.read_csv(os.path.join(args.mount_point, args.csv))
    else:
        df_test = pd.read_parquet(os.path.join(args.mount_point, args.csv))

    test_ds = USDataset(df_test, img_column=args.img_column, transform=test_transform)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4, persistent_workers=True)

    with torch.no_grad():

        features = []
        for idx, X in tqdm(enumerate(test_loader), total=len(test_loader)):
            X = X.cuda(non_blocking=True)
            X_hat, feat = model(X)
            features.append(feat.cpu())
            # if args.nn == "effnet_decode":
            #     X_hat, feat = model(X)
            #     features.append(feat.cpu())
            # else:
            #     feat = model(X)
            #     features.append(feat.cpu())

        features = torch.cat(features, dim=0)
        features = features.numpy()


    out_dir = os.path.join(args.mount_point, args.out, os.path.splitext(os.path.basename(args.model))[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    ext = os.path.splitext(args.csv)[1]

    pickle.dump(features, open(os.path.join(out_dir, os.path.basename(args.csv).replace(ext, "_prediction.pickle")), 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='CSV file for testing', required=True)
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--img_column', type=str, help='Column name in CSV file for image path', default='img_path')
    parser.add_argument('--model', type=str, help='Trained model', required=True)
    parser.add_argument('--nn', help='Type of neural network', type=str, default="monai_autoencoder")
    parser.add_argument('--out', type=str, help='Output filename', default="out.pickle")
    parser.add_argument('--batch_size', type=int, help='Batch size', default=256)
    parser.add_argument('--num_workers', type=int, help='Num workers for data loading', default=8)
    parser.add_argument('--k', type=int, help='Number of top k frames, used for training and reloading', default=128)
    parser.add_argument('--output_dim', help='Output dimension, used for the model effnet_simclr', type=int, default=128)

    args = parser.parse_args()

    main(args)
