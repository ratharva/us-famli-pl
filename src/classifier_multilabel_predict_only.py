import argparse
import itertools
import math
import os
import pandas as pd
import numpy as np 
from torch import nn

import torch
from torch.utils.data import DataLoader

from loaders.ultrasound_dataset_classification import USDataset
from transforms.ultrasound_transforms import USClassEvalTransforms
# from nets.classification_old import EfficientNet
from transferModel import EfficientNetTransfer

from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, roc_curve, auc

from tqdm import tqdm
import scikitplot as skplt
import matplotlib.pyplot as plt

import pickle
import seaborn as sns

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from itertools import cycle

def main(args):

    myLabels = ['No structures visible', 'Head Visible',
       'Abdomen Visible', 'Amniotic fluid visible', 
       'Placenta visible']#, 'Fetus or CRL visible'] 
    
    
    model = EfficientNetTransfer.load_from_checkpoint(args.model)
    model.eval()
    model.cuda()

    if args.extract_features:
        model.extract_features = True

    if(os.path.splitext(args.csv)[1] == ".csv"):        
        df_test = pd.read_csv(os.path.join(args.mount_point, args.csv))
    else:        
        df_test = pd.read_parquet(os.path.join(args.mount_point, args.csv))
    
    test_ds = USDataset(df_test, label_column = None, img_column=args.img_column, transform=USClassEvalTransforms())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)

    with torch.no_grad():
        thresholds = [0.39351434, 0.105817206, 0.06510856, 0.21056508, 0.08419113]
        predictions = []
        y_true = []
        probs = []
        features = []
        accuracyList = []
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        # print("PBAR is: ", pbar)
        for idx, X in pbar: 
            # print("X is: ", X)
            # if use_class_column:
            # print(X.shape)
            # X, Y = X
            X = X.cuda().contiguous()   
            # if args.extract_features:        
            #     pred, x_f = model(X)    
            #     features.append(x_f.cpu().numpy())
            # else:
            #     pred = model(X)
            myPred = model(X)
            myPredSigmoid = nn.Softmax(dim=1)(myPred)
            converted_tensor = torch.where(myPredSigmoid >= torch.tensor(thresholds).unsqueeze(0).cuda(), torch.tensor(1), torch.tensor(0))
            converted_tensor = converted_tensor.cuda()
            predictions.append(converted_tensor.cpu().numpy()[0])
        print("PROBAB SHAPE: ", len(probs))
        # print("Y shape:", Y)
        predDf = pd.DataFrame()
        for label in myLabels:
            label1 = label + "_predicted"
            predDf[label1] = [pred[myLabels.index(label)] for pred in predictions]
        print(predDf.shape)
        print(df_test.shape)
        dfCombined = pd.concat([df_test, predDf], axis=1)
        dfCombined.to_csv("/mnt/raid/home/ayrisbud/us-famli-pl/src/abdomenFemurPredicted.csv", index=False)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Classification predict')
    parser.add_argument('--csv', type=str, help='CSV file for testing', required=True)
    parser.add_argument('--extract_features', type=int, help='Extract the features', default=0)
    parser.add_argument('--img_column', type=str, help='Column name in the csv file with image path', default="uuid_path")
    parser.add_argument('--class_column', type=str, help='Column name in the csv file with classes', default=None)
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--model', help='Model path to continue training', type=str, default=None)
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--out', help='Output directory', type=str, default="./")
    parser.add_argument('--pred_column', help='Output column name', type=str, default="pred_class")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=256)
    parser.add_argument('--nn', help='Type of neural network', type=str, default="efficientnet_b0")

    args = parser.parse_args()

    main(args)
