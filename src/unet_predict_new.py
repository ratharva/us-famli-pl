import argparse
import itertools
import math
import os
import pandas as pd
import numpy as np 
from torch import nn
import copy

import torch
from torch.utils.data import DataLoader

from loaders.ultrasound_dataset_classification import USDataset
from transforms.ultrasound_transforms import USClassEvalTransforms
# from nets.classification_old import EfficientNet
from transferModel import EfficientNetTransfer

from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, roc_curve, auc, precision_recall_curve

from tqdm import tqdm
import scikitplot as skplt
import matplotlib.pyplot as plt

import pickle
import seaborn as sns

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from itertools import cycle
from cloneModelArchitectureV3 import EfficientNetClone
# from cloneModelArchitectureV3Updated import EfficientNetClone
# from multilabelModel import EfficientNetNew
from transferModel import EfficientNetTransfer
# from UnetModelArchitectureMonai import SharedBackboneUNet
from UnetModelArchitectureSimplifiedV2 import SharedBackboneUNet
import datetime


def main(args):

    # myLabels = ['No structures visible', 'Head Visible',
    #    'Abdomen Visible', 'Chest Visible', 'Femur Visible',
    #    'Other arm or leg bones visible', 'Umbilical cord visible',
    #    'Amniotic fluid visible', 'Placenta visible', 'Fetus or CRL visible',
    #    'Maternal bladder visible', 'Head Measurable', 'Abdomen Measurable',
    #    'Femur Measurable']

    # myLabels = ['No structures visible', 'Head Visible',
    #    'Abdomen Visible', 'Amniotic fluid visible', 
    #    'Placenta visible']#, 'Fetus or CRL visible']
    current_datetime = datetime.datetime.now()

    # Convert the datetime object to a string
    date_time_string = current_datetime.strftime('%Y-%m-%d_%H:%M:%S')
    myLabels = ['No structures visible', 'Head Visible',
       'Abdomen Visible', 'Femur Visible', 
       'Placenta visible'] #, 'Fetus or CRL visible' 
    
    
    # model = EfficientNetTransfer.load_from_checkpoint(args.model)
    model = SharedBackboneUNet.load_from_checkpoint(args.model, strict=False)

    # model = EfficientNetClone.load_from_checkpoint(args.model, strict=False)
    # model = EfficientNetNew.load_from_checkpoint(args.model, strict=False)
    model.eval()
    model.cuda()

    if args.extract_features:
        model.extract_features = True

    if(os.path.splitext(args.csv)[1] == ".csv"):        
        df_test = pd.read_csv(os.path.join(args.mount_point, args.csv))
    else:        
        df_test = pd.read_parquet(os.path.join(args.mount_point, args.csv))
    
    test_ds = USDataset(df_test, label_column = myLabels, img_column=args.img_column, transform=USClassEvalTransforms(), repeat_channel=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)


    # def plot_confusion_matrix(cm, classes,
    #                         normalize=False,
    #                         title='Confusion matrix',
    #                         cmap=plt.cm.Blues):
    #     """
    #     This function prints and plots the confusion matrix.
    #     Normalization can be applied by setting `normalize=True`.
    #     """
    #     if normalize:
    #         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #         print("Normalized confusion matrix, avg:", np.trace(cm)/len(classes))
    #     else:
    #         print('Confusion matrix, without normalization')

    #     plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    #     plt.title(title)
    #     plt.colorbar()
    #     tick_marks = np.arange(len(classes))
    #     plt.xticks(tick_marks, classes, rotation=45)
    #     plt.yticks(tick_marks, classes)

    #     fmt = '.3f' if normalize else 'd'
    #     thresh = cm.max() / 2.
    #     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #         plt.text(j, i, format(cm[i, j], fmt),
    #                 horizontalalignment="center",
    #                 color="white" if cm[i, j] > thresh else "black")

    #     plt.xlabel('Predicted label')
    #     plt.ylabel('True label')
    #     plt.tight_layout()

    #     return cm
    # use_class_column = False
    # if args.class_column is not None and args.class_column in df_test.columns:
    #     use_class_column = True

    # if use_class_column:

    #     #unique_classes = np.sort(np.unique(df_test[args.class_column]))    

    #     #class_replace = {}
    #     #for cn, cl in enumerate(unique_classes):
    #     #    class_replace[cl] = cn
    #     #print(unique_classes, class_replace)

    #     #df_test[args.class_column] = df_test[args.class_column].replace(class_replace)

    #     test_ds = USDataset(df_test, img_column=args.img_column, class_column=args.class_column, transform=USClassEvalTransforms())    
    # else:

    #     test_ds = USDataset(df_test, img_column=args.img_column, transform=USClassEvalTransforms())

    # test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)

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
            Y = Y.cuda()
            mySeg, myPred = model(X)
            myPredSigmoid = nn.Softmax(dim=1)(myPred)
            probs.append(myPredSigmoid.cpu().numpy())
            y_true.append(Y.cpu().numpy()[0])


    #CALCULATE ROC STATS
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
        thresh = dict()
        pres = dict()
        rec = dict()
        thre = dict()
        thresholds = []
        secThresh = []
        roc_auc = dict()
        roc_pr = dict()
        for i in range(5):
            fpr[i], tpr[i], thresh[i] = roc_curve(y_true[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            pres[i], rec[i], thre[i] = precision_recall_curve(y_true[:, i], probs[:, i])
            roc_pr[i] = auc(rec[i], pres[i])

        optimal_thresholds = []
        for i in range(5):
            j_statistic = tpr[i] - fpr[i]
            optimal_thresholds.append(thresh[i][np.argmax(j_statistic)])
        secThresh = optimal_thresholds

        optimal_thresholds1 = []
        for i in range(5):
            optimal_thresholds1.append(thre[i][np.argmax(roc_pr[i])])
        secThresh1 = optimal_thresholds1
        
        #     thresholds.append(thresholdss)
        # desired_fpr = 0.01  # Set your desired false positive rate
        # desired_tpr = 0.99  # Set your desired true positive rate
        #   # To store the thresholds for each class
        # secThresh = []
        # for i in range(5):  # Assuming there are 5 classes
        #     # Find the index in the FPR or TPR array that is closest to the desired FPR or TPR
        #     closest_index = np.argmin(np.abs(fpr[i] - desired_fpr))
        #     print(closest_index)
        #     # Use the index to obtain the corresponding threshold
        #     threshold = thresholds[i][closest_index]
        #     secThresh.append(threshold)
        
        # print(secThresh)
        print(secThresh)
        print(secThresh1)

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
        plt.savefig("/mnt/raid/home/ayrisbud/us-famli-pl/src/classification_report/UNet/epoch=51-val_loss=0.382_auc.png")

        segmentationMaps = []
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        y_true = []
        for idx, X in pbar:
            X, Y = X
            X = X.cuda().contiguous()   
            mySeg, myPred = model(X)
            segmentationMaps.append(mySeg)
            myPredSigmoid = nn.Softmax(dim=1)(myPred)
            converted_tensor = torch.where(myPredSigmoid >= torch.tensor(secThresh).unsqueeze(0).cuda(), torch.tensor(1), torch.tensor(0))
            # converted_tensor = torch.where(myPredSigmoid >= 0.08, torch.tensor(1), torch.tensor(0))
            converted_tensor = converted_tensor.cuda()
            Y = Y.cuda()
            correctly_predicted_ones = ((Y == 1) & (converted_tensor == 1)).sum().item()
            total_actual_ones = (Y == 1).sum().item()
            accuracy_for_ones = correctly_predicted_ones / total_actual_ones if total_actual_ones != 0 else 0
            accuracyList.append(accuracy_for_ones)
            predictions.append(converted_tensor.cpu().numpy()[0])
            y_true.append(Y.cpu().numpy()[0])


        torch.save(segmentationMaps, "/mnt/raid/home/ayrisbud/savedSegmentations/" + date_time_string + ".pt")
        print("PROBAB SHAPE: ", len(probs))
        print("Y shape:", Y)
        predDf = pd.DataFrame()
        for label in myLabels:
            label1 = label + "_predicted"
            predDf[label1] = [pred[myLabels.index(label)] for pred in predictions]
        print(predDf.shape)
        print(df_test.shape)
        dfCombined = pd.concat([df_test, predDf], axis=1)
        dfCombined.to_csv("/mnt/raid/home/ayrisbud/us-famli-pl/src/predictedUnetClassifierFullPredSimp.csv", index=False)
        print(dfCombined.columns)
        average_accuracy = sum(accuracyList) / len(accuracyList)
        print(average_accuracy)
        # print(y_true.shape)
        # predictions = np.vstack(predictions)
        # print(predictions.shape)
        myDf = pd.DataFrame({"y_true": y_true, "y_pred": predictions})
        # print(myDf.head())
        print(myDf.shape)
        myDict = classification_report(y_true, predictions, target_names=myLabels, output_dict=True)
        myClassDf = pd.DataFrame(myDict).transpose()
        myClassDf.to_csv("/mnt/raid/home/ayrisbud/us-famli-pl/src/classification_report/UNet/epoch=51-val_loss=0.382.csv", header=True, index=True)
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
        plt.savefig("/mnt/raid/home/ayrisbud/us-famli-pl/src/classification_report/UNet/epoch=51-val_loss=0.382_confMat.png")
        


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
    parser.add_argument('--thresh', help="Set threshold for softmax labels", type=float, default=0.08)


    args = parser.parse_args()

    main(args)
