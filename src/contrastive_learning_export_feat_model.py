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
from nets.contrastive import USMoco, SimCLR, Sim, SimScore
from tqdm import tqdm

from pl_bolts.transforms.dataset_normalizations import (
    imagenet_normalization
)
from torchvision import transforms
from torch.utils.data import DataLoader
import pickle


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
        model = SimScore(emb_dim=args.emb_dim, base_encoder=args.base_encoder).load_from_checkpoint(args.model)
        
        transform = SimTestTransforms(224)

        if hasattr(model.convnet, 'classifier'):
            model.convnet.classifier = nn.Identity()
        elif hasattr(model.convnet, 'fc'):        
            model.convnet.fc = nn.Identity()

        model = model.convnet

    torch.save(model.state_dict(), args.out)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Export features model')
    parser.add_argument('--model', help='Trained model checkpoint', type=str, required=True)
    parser.add_argument('--out', help='Output filename', type=str, default="model.pt")
    parser.add_argument('--nn', help='Type of neural network', type=str, default="moco_v2")
    parser.add_argument('--emb_dim', help='Embedding dimension', type=int, default=128)    
    parser.add_argument('--base_encoder', help='What encoder to use', type=str, default='efficientnet_b0')


    args = parser.parse_args()

    main(args)
