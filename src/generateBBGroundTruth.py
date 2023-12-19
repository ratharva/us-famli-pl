import argparse
from torch.autograd import Function
import math
import os
import pandas as pd
import numpy as np
from scipy.special import softmax
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn

# from loaders.ultrasound_dataset import USDataset
from torch.utils.data import DataLoader

from loaders.ultrasound_dataset_classification import USDataset
from transforms.ultrasound_transforms import USClassEvalTransforms
from transferModel import EfficientNetTransfer
# from transforms.ultrasound_transforms import USEvalTransforms

from sklearn.utils import class_weight
from sklearn.metrics import classification_report

from tqdm import tqdm

import pickle

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from captum.attr import GuidedGradCam, GuidedBackprop
import matplotlib.pyplot as plt
plt.ioff()
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from torchvision import transforms
from monai.transforms import ScaleIntensityRange

from pl_bolts.transforms.dataset_normalizations import (
    imagenet_normalization
)
import nrrd
from PIL import Image

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import torchvision.transforms as T
from PIL import Image
import cv2
# from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import pydicom
from sklearn.model_selection import train_test_split
import uuid
import copy
# %matplotlib inline


def figArr(fig, draw=True):
    fig.set_facecolor("black")
    myCanvas = fig.canvas#FigureCanvas(fig)
    myCanvas.draw()
    w, h = myCanvas.get_width_height()#fig.get_size_inches() * fig.get_dpi()
    myArr = np.frombuffer(myCanvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    myImg = cv2.cvtColor(myArr, cv2.COLOR_RGB2BGR)
    # plt_fig.grid()
    return myImg

def useGradcam(myCSV, myModel, myImgCol, myLabels, myNn, myOutFile, myBatchSize, myNumWorkers, myMountPoint, myExtractFeatures):
    model = EfficientNetTransfer(base_encoder=myNn, ckpt_path="/mnt/raid/C1_ML_Analysis/train_output/classification/extract_frames_blind_sweeps_c1_30082022_wscores_train_train_sample_clean_feat/epoch=9-val_loss=0.27.ckpt").load_from_checkpoint(myModel)
    model.eval()
    model.cuda()
    # print(model)
    myGuidedGradCam = GuidedGradCam(model, model.efficientnet.convnet.features[8][0])

    myLabelList = ['No structures visible', 'Head Visible',
                'Abdomen Visible', 'Amniotic fluid visible', 
                'Placenta visible', 'Fetus or CRL visible']
    if myExtractFeatures:
            model.extract_features = True

    if(os.path.splitext(myCSV)[1] == ".csv"):        
        df_test = pd.read_csv(os.path.join(myMountPoint, myCSV))
    else:        
        df_test = pd.read_parquet(os.path.join(myMountPoint, myCSV))
        
    test_ds = USDataset(df_test, label_column = None, img_column=myImgCol, transform=USClassEvalTransforms(), mount_point=myMountPoint)
    test_loader = DataLoader(test_ds, batch_size=myBatchSize, shuffle=False, num_workers=myNumWorkers, pin_memory=True, prefetch_factor=4)

    transform = transforms.Compose([

    transforms.CenterCrop(256),
    # ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),

    ])
    predictions = []
    probs = []
    features = []
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for idx, X in pbar:
        if myLabels:
            imgTransform = transform(X)
        # print(X)
        X = X.cuda().contiguous()   
        if myExtractFeatures:
            # print("Does it enter this if2?")        
            pred, x_f = model(X)    
            features.append(x_f.cpu().numpy())
        else:
            # print("Does it enter this else?")
            pred = model(X)
        myPredSigmoid = nn.Softmax(dim=1)(pred)
        converted_tensor = torch.where(myPredSigmoid >= 0.08, torch.tensor(1), torch.tensor(0))
        converted_tensor = converted_tensor.cuda()
        converted_tensor = np.array(converted_tensor.cpu())
        # print("Converted Tensor", converted_tensor)

        isInArray = np.any(converted_tensor == 1)
        # print(isInArray)
        if isInArray:
            oriImag = np.transpose(X.squeeze().cpu().detach().numpy(), (1,2,0))
            # print("OriImageShape: ", oriImag.shape)

            fig = plt.figure(figsize=(6, 6))

            # Add a subplot to the figure
            ax = fig.add_subplot(111)

            # Display the oriImag using imshow on the subplot
            img_arr = ax.imshow(oriImag)
            originalImage = figArr(fig)
            tempImage = copy.deepcopy(originalImage)
            myDict = {}
            myYoloList = []
        # print("Length of convertedArray: ", len(converted_tensor[0]))
        # converted_tensor[0] = converted_tensor[0][1:]
        myUUID = uuid.uuid4()
        for i in range(len(converted_tensor[0])):
            # print(type(converted_tensor[0][i]))
            if converted_tensor[0][0] == 1:
                break
            
            if converted_tensor[0][i] == 1:
                # print("i is: ", i)
                # print("Class is: ", myLabelList[i])
                myImageAttributes = myGuidedGradCam.attribute(X, i, interpolate_mode="bilinear")
                if np.count_nonzero(myImageAttributes) == 0:
                    break
                # print("Image Attributes are: ", np.count_nonzero(myImageAttributes))
                # cv2.imwrite("/mnt/raid/home/ayrisbud/USOD/images/train/whatImage.png", tempImage)
                # print(myImageAttributes.shape)
                default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                    [(0, '#ffffff'),
                                                    (0.25, '#000000'),
                                                    (1, '#000000')], N=256)

                myPlt = viz.visualize_image_attr(np.transpose(myImageAttributes.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(imgTransform.squeeze().cpu().detach().numpy(), (1,2,0)), 
                                        "heat_map",
                                        cmap="magma",
                                        # sign="absolute_value",
                                        #   show_colorbar=True,
                                        # fig_size=(3.56,3.56),
                                        use_pyplot=True
                                        )
                
                # fig, axis = plt.subplots(1, 4, figsize=(30,30))

                myCv2Img = figArr(myPlt[0])
                # cv2.imwrite("./tempImagesPres/Heatmap" + str(idx) + ".jpg", myCv2Img)
                # originalImage = figArr(myPlt1[0])
                # cv2.imwrite("./tempImagesPres/Original" + str(idx) + ".jpg", originalImage)
                kernel = np.ones((5, 5), np.uint8)
                # dialatedImg = cv2.dilate(myCv2Img, kernel, iterations=1)
                # plt.axis('off')
                # axis[0].imshow(cv2.cvtColor(myCv2Img, cv2.COLOR_BGR2RGB))
                myGrayImg = cv2.cvtColor(myCv2Img, cv2.COLOR_BGR2GRAY)
                myThresh = cv2.threshold(myGrayImg, 30, 255, cv2.THRESH_BINARY)[1]  #+ cv2.THRESH_OTSU
                dialatedImg = cv2.dilate(myThresh, kernel, iterations=1)
                # plt.axis('off')
                # axis[1].imshow(cv2.cvtColor(dialatedImg, cv2.COLOR_BGR2RGB))
                # cv2.imwrite("./tempImagesPres/dialated" + str(idx) + ".jpg", dialatedImg)
                imgCopy = originalImage.copy()
                imgCopy1 = myCv2Img.copy()
                # edged = cv2.Canny(dialatedImg, 30, 200)
                myContours, myHierearchy = cv2.findContours(dialatedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                sortedConts = sorted(myContours, key=cv2.contourArea, reverse=True)
                largestContours = sortedConts[0:1]
                # myContours = myContours[0] if len(myContours) == 2 else myContours[1]
                contDraw = cv2.drawContours(dialatedImg, largestContours, -1, (255, 255, 0), 3)
                # axis[2].imshow(cv2.cvtColor(contDraw, cv2.COLOR_BGR2RGB))
                for cntr in largestContours:
                    if cv2.arcLength(cntr, True) > 300:
                        # print("ARCLENGTH", cv2.arcLength(cntr, True))
                        x,y,w,h = cv2.boundingRect(cntr)
                        cv2.rectangle(originalImage, (x, y), (x+w, y+h), (36, 255, 12), 2)
                        x2 = x + w
                        y2 = y + h
                        xc = (x + x2) / 2
                        yc = (y + y2) / 2
                        nxc = xc / originalImage.shape[0]
                        nyc = yc / originalImage.shape[1]
                        nw = w / 600
                        nh = h / 600
                        datFormat = f"{i} {nxc} {nyc} {nw} {nh}"
                        # print("YOLO FormatData: ", datFormat)
                        myYoloList.append(datFormat)
                        myDict[myLabelList[i]] = (nxc, nyc, nw, nh)
                    # print("x,y,w,h:",x,y,w,h)
                for cntr in largestContours:
                    if cv2.arcLength(cntr, True) > 300:
                        # print("ARCLENGTH", cv2.arcLength(cntr, True))
                        x,y,w,h = cv2.boundingRect(cntr)
                        cv2.rectangle(imgCopy1, (x, y), (x+w, y+h), (36, 255, 12), 2)
                        # print("x,y,w,h:",x,y,w,h)
                # plt.axis('off')
                # axis[2].imshow(cv2.cvtColor(imgCopy1, cv2.COLOR_BGR2RGB))
                # cv2.imwrite("./tempImagesPres/heatmapBox" + str(idx) + ".jpg", imgCopy1)
                # plt.axis('off')
                # axis[3].imshow(cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB))
                # tempImage = originalImage
                # cv2.imwrite("./tempImagesPres/OriginalBox" + str(idx) + ".jpg", imgCopy)

                # head, tail = os.path.split(img_path[0])
                # if not os.path.isdir("./gradCamImages/" + head):
                #     os.makedirs("./gradCamImages/" + head)
            else:
                print("No Class found!")
        # print(myDict)
        # print(myYoloList)
        # plt.figure(2, figsize=(6,6))
        # print("ORI IMG DIM: ", originalImage.shape)
        # plt.imshow(cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB))
        # print("intermediate")
        # plt.axis('off')
        # plt.show()
        # print("afterPlot")
        # plt.figure(3, figsize=(6,6))
        # print("ORI IMG DIM: ", tempImage.shape)
        # plt.imshow(cv2.cvtColor(tempImage, cv2.COLOR_BGR2RGB))
        # print("intermediate")
        # plt.axis('off')
        # plt.show()
        # print("afterPlot")
        cv2.imwrite("/mnt/raid/home/ayrisbud/USOD/images/val/" + str(myUUID) + ".png", tempImage)
        pathToFile = "/mnt/raid/home/ayrisbud/USOD/labels/val/" + str(myUUID) + ".txt"
        file = open(pathToFile, "w+")
        for item in myYoloList:
            file.write(item + "\n")
        file.close()
        
        # if idx == 2:
        #     break

        
if __name__ == '__main__':

    myCSV = "/mnt/raid/home/ayrisbud/us-famli-pl/src/annotatedValConcise.csv"
    # myModel = "/mnt/raid/home/ayrisbud/train_output/classification/epoch=21-val_loss=1.06.ckpt"
    # myModel = "/mnt/raid/home/ayrisbud/train_output/classification/epoch=35-val_loss=1.01.ckpt"
    myModel = "/mnt/raid/home/ayrisbud/train_output/classification/clone/epoch=100-val_loss=0.14.ckpt"
    myImgCol = "img_path"
    myClassCol = "pred_cluster"
    myNn = "efficientnet_b0"    
    myOutFile = "./myOutput"
    myBatchSize = 1
    myNumWorkers = 16
    myMountPoint = "/mnt/raid/C1_ML_Analysis/"
    myExtractFeatures = False
    myLabels = True
    

    # main(myCSV, myModel, myImgCol, myClassCol, myNn, myOutFile, myBatchSize, myNumWorkers, myMountPoint, myExtractFeatures)
    useGradcam(myCSV=myCSV, myModel=myModel, myImgCol = myImgCol, myNn=myNn, myOutFile=myOutFile, myBatchSize=myBatchSize, myNumWorkers=myNumWorkers, myMountPoint=myMountPoint, myExtractFeatures=myExtractFeatures, myLabels=myLabels)
    # useGBackprop(myCSV, myModel, myImgCol, myClassCol, myNn, myOutFile, myBatchSize, myNumWorkers, myMountPoint, myExtractFeatures)
    # getBoundingImages(myCSV, myModel, myImgCol, myClassCol, myNn, myOutFile, myBatchSize, myNumWorkers, myMountPoint, myExtractFeatures)