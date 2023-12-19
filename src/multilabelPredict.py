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
# from multilabelModel import EfficientNetNew
# from cloneModelArchitectureV3 import EfficientNetClone
from transferModel import EfficientNetTransfer
def main(args):

    # myLabels = ['No structures visible', 'Head Visible',
    #    'Abdomen Visible', 'Chest Visible', 'Femur Visible',
    #    'Other arm or leg bones visible', 'Umbilical cord visible',
    #    'Amniotic fluid visible', 'Placenta visible', 'Fetus or CRL visible',
    #    'Maternal bladder visible', 'Head Measurable', 'Abdomen Measurable',
    #    'Femur Measurable']

    myLabels = ['No structures visible', 'Head Visible',
       'Abdomen Visible', 'Amniotic fluid visible', 
       'Placenta visible'] #, 'Fetus or CRL visible' 

    # myLabels = ['No structures visible', 'Head Visible',
    #    'Abdomen Visible', 'Femur Visible', 
    #    'Placenta visible'] #, 'Fetus or CRL visible' 

    # modelFile = "/mnt/raid/home/ayrisbud/train_output/classification/epoch=42-val_loss=0.58.ckpt"

    
    # model = EfficientNetClone.load_from_checkpoint(args.model, strict=False)
    model = EfficientNetTransfer.load_from_checkpoint(args.model, strict=False)
    model.eval()
    model.cuda()

    if args.extract_features:
        model.extract_features = True

    if(os.path.splitext(args.csv)[1] == ".csv"):        
        df_test = pd.read_csv(os.path.join(args.mount_point, args.csv))
    else:        
        df_test = pd.read_parquet(os.path.join(args.mount_point, args.csv))
    
    test_ds = USDataset(df_test, label_column = myLabels, img_column=args.img_column, transform=USClassEvalTransforms())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)

    with torch.no_grad():

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
            X, Y = X
            X = X.cuda().contiguous()   
            myPred = model(X)
            myPredSigmoid = nn.Sigmoid()(myPred)
            # print(myPredSigmoid.unique())
            # print(myPredSigmoid)
            # print("Actual: ", Y)
            threshold = 0.8
            predicted = (myPredSigmoid >= threshold).float()
            # print("Predicted: ", predicted)
            # if idx > 10:
            #     break
            # myPredSigmoid = nn.Softmax(dim=1)(myPred)
            # converted_tensor = torch.where(myPredSigmoid >= 0.08, torch.tensor(1), torch.tensor(0))
            # converted_tensor = converted_tensor.cuda()
            Y = Y.cuda()
            correctly_predicted_ones = ((Y == 1) & (predicted == 1)).sum().item()
            # print("Correctly predicted ones: ", correctly_predicted_ones)
            total_actual_ones = (Y == 1).sum().item()
            # print(converted_tensor)
            # print(Y)
            # print("Actual ones: ", total_actual_ones)
            accuracy_for_ones = correctly_predicted_ones / total_actual_ones if total_actual_ones != 0 else 0
            # print("Accuracy is: ", accuracy_for_ones)
            accuracyList.append(accuracy_for_ones)
            predictions.append(predicted.cpu().numpy()[0])
            y_true.append(Y.cpu().numpy()[0])
            probs.append(myPred.cpu().numpy())
        print("PROBAB SHAPE: ", len(probs))
        print("Y shape:", Y)
        predDf = pd.DataFrame()
        for label in myLabels:
            label1 = label + "_predicted"
            predDf[label1] = [pred[myLabels.index(label)] for pred in predictions]
        print(predDf.shape)
        print(df_test.shape)
        dfCombined = pd.concat([df_test, predDf], axis=1)
        dfCombined.to_csv("/mnt/raid/home/ayrisbud/us-famli-pl/src/predictedClassifierOrigDat.csv", index=False)
        print(dfCombined.columns)
        average_accuracy = sum(accuracyList) / len(accuracyList)
        print(average_accuracy)
        myDf = pd.DataFrame({"y_true": y_true, "y_pred": predictions})
        # print(myDf.head())
        print(myDf.shape)
        myDict = classification_report(y_true, predictions, target_names=myLabels, output_dict=True)
        myClassDf = pd.DataFrame(myDict).transpose()
        myClassDf.to_csv("/mnt/raid/home/ayrisbud/us-famli-pl/src/classification_report/epoch=17-val_loss=0.173.csv", header=True, index=True)
        print(myClassDf)
        myConfMat = multilabel_confusion_matrix(y_true, predictions)
        # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        normalized_confusion_matrix = np.array([
            # row / row.sum() if row.sum() > 0 else row
            row.astype('float') / row.sum(axis=1)[:, np.newaxis] if row.sum() > 0 else row
            for row in myConfMat
        ])
        print("MYCONFMAT: ", myConfMat.shape, type(myConfMat))
        print("NORMMAT: ", normalized_confusion_matrix.shape, type(normalized_confusion_matrix))
        # disp = ConfusionMatrixDisplay(confusion_matrix=myConfMat, display_labels=myLabels)
        # disp.plot()
        plt.figure(figsize=(12, 8))
        plt.figure(figsize=(12, 8))
        for i, cm in enumerate(normalized_confusion_matrix):
            plt.subplot(2, 3, i + 1)
            # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            #             xticklabels=['False', 'True'], yticklabels=['False', 'True'])
            sns.heatmap(cm, annot=True, cmap='Blues', cbar=False,
                        xticklabels=['False', 'True'], yticklabels=['False', 'True'])
            plt.title(f'Class {i}: {myLabels[i]}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        plt.savefig("/mnt/raid/home/ayrisbud/us-famli-pl/src/classification_report/epoch=17-val_loss=0.173_confMat.png")
        # skplt.metrics.plot_roc_curve(y_true, predictions)
        # plt.savefig("/mnt/raid/home/ayrisbud/us-famli-pl/src/classification_report/epoch=35-val_loss=1.01_auc.png")
        predictions = np.array(predictions)
        probs = np.array(probs)
        probs = probs.reshape(1805, 5)
        myProbUinque = np.unique(probs, return_counts = True)
        print(myProbUinque)
        print(np.max(probs))
        print(np.min(probs))
        print(np.sum(probs > 1.5))
        print(np.sum(probs < 1.5))
        print(np.sum(probs == 0))
       
        # print(np.mode(probs))
        y_true = np.array(y_true)
        print(np.sum(y_true > 0))
        print(np.sum(y_true == 0))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(5):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot the ROC curves
        plt.figure(figsize=(10, 8))
        colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow'])
        for i, color in zip(range(5), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label= myLabels[i] + ' (AUC = {0:0.2f})'.format(roc_auc[i]))

        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multilabel ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig("/mnt/raid/home/ayrisbud/us-famli-pl/src/classification_report/epoch=17-val_loss=0.173_auc.png")


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
