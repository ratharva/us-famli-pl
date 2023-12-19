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
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, roc_curve, auc

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

def main(args):

    # myLabels = ['No structures visible', 'Head Visible',
    #    'Abdomen Visible', 'Chest Visible', 'Femur Visible',
    #    'Other arm or leg bones visible', 'Umbilical cord visible',
    #    'Amniotic fluid visible', 'Placenta visible', 'Fetus or CRL visible',
    #    'Maternal bladder visible', 'Head Measurable', 'Abdomen Measurable',
    #    'Femur Measurable']

    myLabels = ['No structures visible', 'Head Visible',
       'Abdomen Visible', 'Amniotic fluid visible', 
       'Placenta visible']#, 'Fetus or CRL visible']

    # myLabels = ['No structures visible', 'Head Visible',
    #    'Abdomen Visible', 'Femur Visible', 
    #    'Placenta visible'] #, 'Fetus or CRL visible' 
    
    
    model = EfficientNetTransfer.load_from_checkpoint(args.model)
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
    
    test_ds = USDataset(df_test, label_column = myLabels, img_column=args.img_column, transform=USClassEvalTransforms())
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
            # if args.extract_features:        
            #     pred, x_f = model(X)    
            #     features.append(x_f.cpu().numpy())
            # else:
            #     pred = model(X)
            myPred = model(X)
            myPredSigmoid = nn.Softmax(dim=1)(myPred)
            converted_tensor = torch.where(myPredSigmoid >= args.thresh, torch.tensor(1), torch.tensor(0))
            converted_tensor = converted_tensor.cuda()
            Y = Y.cuda()
            correctly_predicted_ones = ((Y == 1) & (converted_tensor == 1)).sum().item()
            # print("Correctly predicted ones: ", correctly_predicted_ones)
            total_actual_ones = (Y == 1).sum().item()
            # print(converted_tensor)
            # print(Y)
            # print("Actual ones: ", total_actual_ones)
            accuracy_for_ones = correctly_predicted_ones / total_actual_ones if total_actual_ones != 0 else 0
            # print("Accuracy is: ", accuracy_for_ones)
            accuracyList.append(accuracy_for_ones)
            predictions.append(converted_tensor.cpu().numpy()[0])
            y_true.append(Y.cpu().numpy()[0])
            probs.append(myPredSigmoid.cpu().numpy())

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
        print(y_true)
        print(predictions)
        raise NotImplementedError
        myDf = pd.DataFrame({"y_true": y_true, "y_pred": predictions})
        # print(myDf.head())
        print(myDf.shape)
        myDict = classification_report(y_true, predictions, target_names=myLabels, output_dict=True)
        myClassDf = pd.DataFrame(myDict).transpose()
        myClassDf.to_csv("/mnt/raid/home/ayrisbud/us-famli-pl/src/classification_report/epoch=54-val_loss=0.272.csv", header=True, index=True)
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
        plt.savefig("/mnt/raid/home/ayrisbud/us-famli-pl/src/classification_report/epoch=54-val_loss=0.272_confMat.png")
        # skplt.metrics.plot_roc_curve(y_true, predictions)
        # plt.savefig("/mnt/raid/home/ayrisbud/us-famli-pl/src/classification_report/epoch=35-val_loss=1.01_auc.png")
        predictions = np.array(predictions)
        probs = np.array(probs)
        probs = probs.reshape(700, 5)
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
        thresholds = []
        roc_auc = dict()
        for i in range(5):
            # print(predictions[:])
            # print(i)
            # print(probs.shape)
            # print(probs[:, i])
            # print(y_true[:])
            
            
            # print(len(y_true))
            # print(y_true[0].shape)
            # print(len(predictions))
            # print(predictions[0].shape)
            fpr[i], tpr[i], thresholdss = roc_curve(y_true[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            # print("Threshold ", str(i), ": ",thresholdss)
            thresholds.append(thresholdss)

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
        plt.savefig("/mnt/raid/home/ayrisbud/us-famli-pl/src/classification_report/epoch=54-val_loss=0.272_auc.png")
           
            #CALCULATE ROC STATS
        probs = np.array(probs)
        probs = probs.reshape(700, 5)
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
        thresholds = []
        roc_auc = dict()
        for i in range(5):
            # print(predictions[:])
            # print(i)
            # print(probs.shape)
            # print(probs[:, i])
            # print(y_true[:])
            
            
            # print(len(y_true))
            # print(y_true[0].shape)
            # print(len(predictions))
            # print(predictions[0].shape)
            fpr[i], tpr[i], thresholdss = roc_curve(y_true[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            # print("Threshold ", str(i), ": ",thresholdss)
            thresholds.append(thresholdss)
        desired_fpr = 0.01  # Set your desired false positive rate
        desired_tpr = 0.99  # Set your desired true positive rate
          # To store the thresholds for each class
        secThresh = []
        for i in range(5):  # Assuming there are 5 classes
            # Find the index in the FPR or TPR array that is closest to the desired FPR or TPR
            closest_index = np.argmin(np.abs(fpr[i] - desired_fpr))
            print(closest_index)
            # Use the index to obtain the corresponding threshold
            threshold = thresholds[i][closest_index]
            secThresh.append(threshold)
        
        print(secThresh)
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
        plt.savefig("/mnt/raid/home/ayrisbud/us-famli-pl/src/classification_report/epoch=54-val_loss=0.272_auc.png")

        # print(myClassDf.head())
        # print(classification_report(y_true, predictions, target_names=myLabels))

    #     dfPredicted = pd.DataFrame(predictions, columns=myLabels)
    #     dfTrue = pd.DataFrame(y_true, columns=myLabels)
    #     dfPredicted["Head Visible"] = dfPredicted['Head Visible'] | dfPredicted['Head Measurable']
    #     dfPredicted["Abdomen Visible"] = dfPredicted['Abdomen Visible'] | dfPredicted['Abdomen Measurable']
    #     dfPredicted["Femur Visible"] = dfPredicted['Femur Visible'] | dfPredicted['Femur Measurable']


    #     dfTrue["Head Visible"] = dfTrue['Head Visible'] | dfTrue['Head Measurable']
    #     dfTrue["Abdomen Visible"] = dfTrue['Abdomen Visible'] | dfTrue['Abdomen Measurable']
    #     dfTrue["Femur Visible"] = dfTrue['Femur Visible'] | dfTrue['Femur Measurable']

    #     columns_to_drop = ['Head Measurable', 'Abdomen Measurable', 'Femur Measurable']
    #     dfPredicted = dfPredicted.drop(columns=columns_to_drop, axis=1)
    #     dfTrue = dfTrue.drop(columns=columns_to_drop, axis=1)

    #     y_g = dfTrue.values
    #     y_p = dfPredicted.values

    #     myLabels1 = ['No structures visible', 'Head Visible',
    #    'Abdomen Visible', 'Chest Visible', 'Femur Visible',
    #    'Other arm or leg bones visible', 'Umbilical cord visible',
    #    'Amniotic fluid visible', 'Placenta visible', 'Fetus or CRL visible',
    #    'Maternal bladder visible']

    #     print(classification_report(y_g, y_p, target_names=myLabels1))

        # y_test = myDf.
        # print(classification_report(y_true, predictions, target_names=myLabels))
            # probs.append(pred.cpu().numpy())        
            # pred = torch.argmax(pred, dim=1).cpu().numpy()
            # # pbar.set_description("prediction: {pred}".format(pred=pred))
            # predictions.append(pred)
            

    # df_test[args.pred_column] = np.concatenate(predictions, axis=0)
    # probs = np.concatenate(probs, axis=0)    


    # out_dir = os.path.join(args.out, os.path.splitext(os.path.basename(args.model))[0])
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)


    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
        
    # # if use_class_column:
    # #     print(classification_report(df_test[args.class_column], df_test[args.pred_column]))

    # ext = os.path.splitext(args.csv)[1]
    # if(ext == ".csv"):
    #     df_test.to_csv(os.path.join(args.mount_point, out_dir, os.path.basename(args.csv).replace(".csv", "_prediction.csv")), index=False)
    # else:        
    #     df_test.to_parquet(os.path.join(args.mount_point, out_dir, os.path.basename(args.csv).replace(".parquet", "_prediction.parquet")), index=False)

    

    # pickle.dump(probs, open(os.path.join(args.mount_point, out_dir, os.path.basename(args.csv).replace(ext, "_probs.pickle")), 'wb'))

    # if len(features) > 0:
    #     features = np.concatenate(features, axis=0)
    #     pickle.dump(features, open(os.path.join(args.mount_point, out_dir, os.path.basename(args.csv).replace(ext, "_prediction.pickle")), 'wb'))


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