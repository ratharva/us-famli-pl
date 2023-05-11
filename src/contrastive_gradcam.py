import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch import nn 
from loaders.ultrasound_dataset import USDataset
from transforms.ultrasound_transforms import Moco2TestTransforms, SimCLRTestTransforms, SimTestTransforms
# from pl_bolts.models.self_supervised import Moco_v2
from nets.contrastive import USMoco, SimCLR, Sim, SimScore, SimNorth
from nets.ga_net import TimeDistributed
from tqdm import tqdm

from pl_bolts.transforms.dataset_normalizations import (
    imagenet_normalization
)
from torchvision import transforms
from torch.utils.data import DataLoader
import pickle

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import cv2
from PIL import Image

from monai.transforms import (    
    ScaleIntensityRange
)

import nrrd

def main(args):

    transform = None
    if args.nn == "moco_v2":
        model = USMoco(base_encoder=args.base_encoder, emb_dim=args.emb_dim).load_from_checkpoint(args.model)
        model = model.encoder_q.eval()
        transform = SimTestTransforms(224)

    elif args.nn == "simclr":
        model = SimCLR(hidden_dim=args.emb_dim, base_encoder=args.base_encoder).load_from_checkpoint(args.model)
        
        transform = SimCLRTestTransforms(224)

    elif args.nn == "sim":
        model = Sim(emb_dim=args.emb_dim, base_encoder=args.base_encoder).load_from_checkpoint(args.model)
        
        transform = SimTestTransforms(224)
    elif args.nn == "simscore":
        model = SimScore(emb_dim=args.emb_dim, base_encoder=args.base_encoder, hidden_dim=args.hidden_dim).load_from_checkpoint(args.model)
    elif args.nn == "simnorth":
        model = SimNorth(args).load_from_checkpoint(args.model)
        
        transform = SimTestTransforms(224)

    # if args.compute_feat:
    #     if hasattr(model, 'classifier'):
    #         model.classifier = nn.Identity()
    #     elif hasattr(model, 'fc'):
    #         model.fc = nn.Identity()

    model.eval()
    model.cuda() 

    target_layers = [model.convnet.features[-1]]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    
    if(os.path.splitext(args.csv)[1] == ".csv"):        
        df_test = pd.read_csv(os.path.join(args.mount_point, args.csv))
    else:        
        df_test = pd.read_parquet(os.path.join(args.mount_point, args.csv))


    if args.query:
        df_test = df_test.query(args.query).reset_index(drop=True)

    test_ds = USDataset(df_test, transform=transform, img_column=args.img_column)    

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)


    scale_intensity = ScaleIntensityRange(0.0, 1.0, 0, 255)

    
    out_dir = os.path.join(args.out, os.path.splitext(os.path.basename(args.model))[0], "grad_cam")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for idx, X in tqdm(enumerate(test_loader), total=len(test_loader)):             
        X = X.cuda(non_blocking=True).contiguous()        

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.

        targets = None
        if args.target_class is not None:
            targets = [ClassifierOutputTarget(args.target_class)]

        gcam_np = cam(input_tensor=X, targets=targets, eigen_smooth=args.eigen_smooth)


        out_fn = os.path.join(out_dir, df_test.loc[idx][args.img_column])

        out_dir_fn = os.path.dirname(out_fn)

        if not os.path.exists(out_dir_fn):
            os.makedirs(out_dir_fn)

        gcam_np = scale_intensity(gcam_np).squeeze().numpy().astype(np.uint8)

        nrrd.write(out_fn, gcam_np, index_order="C")        

        # print(np.min(gcam_np), np.max(gcam_np)) -> between 0, 0.9999

        # vid_path = df_test.loc[idx][args.img_column]

        # out_vid_path = vid_path.replace(os.path.splitext(vid_path)[1], '.avi')

        # out_vid_path = os.path.join(args.out, out_vid_path)

        # out_dir = os.path.dirname(out_vid_path)

        # if not os.path.exists(out_dir):
        #     os.makedirs(out_dir)

        # vid_np = scale_intensity(X).permute(0,1,3,4,2).squeeze().cpu().numpy().squeeze().astype(np.uint8)
        # gcam_np = scale_intensity(gcam_np).squeeze().numpy().astype(np.uint8)

        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # out = cv2.VideoWriter(out_vid_path, fourcc, args.fps, (256, 256))

        # for v, g in zip(vid_np, gcam_np):
        #     c = cv2.applyColorMap(g, cv2.COLORMAP_JET)
        #     b = cv2.addWeighted(v, 0.5, c, 0.5, 0)
        #     out.write(b)

        # out.release()            


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Autoencoder Training')    
    parser.add_argument('--csv', type=str, help='CSV file for testing', required=True)
    parser.add_argument('--query', type=str, help='Apply a query to the dataframe', default=None)
    parser.add_argument('--img_column', type=str, help='CSV file for testing', default="img_path")    
    parser.add_argument('--model', help='Trained model checkpoint', type=str, required=True)
    parser.add_argument('--out', help='Output directory, the model name will be appended as a directory', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=1)
    parser.add_argument('--nn', help='Type of neural network', type=str, default="moco_v2")
    parser.add_argument('--emb_dim', help='Embedding dimension', type=int, default=128)
    parser.add_argument('--hidden_dim', help='Hidden Embedding dimension', type=int, default=None)
    # parser.add_argument('--compute_feat', help='Remove last MLP Layer', type=bool, default=False)
    parser.add_argument('--base_encoder', help='What encoder to use', type=str, default='efficientnet_b0')
    parser.add_argument('--n_clusters', help='Guess number of clusters', type=int, default=64)
    parser.add_argument('--n_lights', help='Light house', type=int, default=10)

    parser.add_argument('--num_frames', help='Number of frames for the prediction', type=int, default=512)
    parser.add_argument('--eigen_smooth', help='Eigen smooth', type=int, default=0)
    parser.add_argument('--target_class', help='Target class', type=int, default=None)
    parser.add_argument('--fps', help='Frames per second', type=int, default=20)


    args = parser.parse_args()

    main(args)

