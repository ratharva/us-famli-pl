# from monai.losses import DiceBCELoss
from typing import List, Union
from monai.losses import DiceLoss
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
import timm
import torchmetrics
import copy
import numpy as np
import sys
import torchvision
from captum.attr import GuidedGradCam, GuidedBackprop
from nets.classification_old import EfficientNet
from pytorch_lightning.loggers import TensorBoardLogger
from numpy import ndarray
from enum import Enum
from typing import Any, Iterable, List, Optional, Tuple, Union
import warnings
import torch.nn.functional as F
from monai.networks.nets import FlexibleUNet, EfficientNetBN
from monai.visualize.class_activation_maps import GradCAMpp
import monai.visualize.class_activation_maps as cam
import monai
from sklearn.metrics import auc
torch.multiprocessing.set_sharing_strategy('file_system')
# from torcheval.metrics import MultilabelAUPRC
# import torcheval.metrics
# from pytorch_lightning.metrics.functional import auc, auroc
# from pytorch_lightning.metrics.classification import AUROC


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1., gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss()(inputs, targets) #, reduction='none'
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha.cuda() * (1 - pt)**self.gamma * CE_loss
        return F_loss.mean()
class VisualizeSign(Enum):
    positive = 1
    absolute_value = 2
    negative = 3
    all = 4

def _cumulative_sum_threshold(values: ndarray, percentile: Union[int, float]):
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]

def _normalize_scale(attr: ndarray, scale_factor: float):
    assert scale_factor != 0, "Cannot normalize by scale factor = 0"
    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results"
            "may be misleading. This likely means that attribution values are all"
            "close to 0."
        )
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)

def _normalize_attr(
    attr: ndarray,
    sign: str,
    outlier_perc: Union[int, float] = 2,
    reduction_axis: Optional[int] = None,
):
    attr_combined = attr
    if reduction_axis is not None:
        attr_combined = np.sum(attr, axis=reduction_axis)

    # Choose appropriate signed values and rescale, removing given outlier percentage.
    if VisualizeSign[sign] == VisualizeSign.all:
        # print("If1")
        threshold = _cumulative_sum_threshold(np.abs(attr_combined), 100 - outlier_perc)
    elif VisualizeSign[sign] == VisualizeSign.positive:
        # print("If2")
        attr_combined = (attr_combined > 0) * attr_combined
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    elif VisualizeSign[sign] == VisualizeSign.negative:
        # print("If3")
        attr_combined = (attr_combined < 0) * attr_combined
        threshold = -1 * _cumulative_sum_threshold(
            np.abs(attr_combined), 100 - outlier_perc
        )
    elif VisualizeSign[sign] == VisualizeSign.absolute_value:
        # print("If4")
        attr_combined = np.abs(attr_combined)
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
        # print("Value of threshold: ", threshold)
    else:
        raise AssertionError("Visualize Sign type is not valid.")
    return _normalize_scale(attr_combined, threshold)

def dice_loss(heatmap1, heatmap2):
    numerator = 2 * np.sum(heatmap1 * heatmap2)
    denominator = np.sum(heatmap1) + np.sum(heatmap2)
    return 1 - numerator / (denominator + 1e-6)

def remove_overlap(heatmaps, threshold=0.5):
    # Create binary masks based on the threshold
    # print("Type of Heatmap", type(heatmaps[0]))
    # print("Shape of heatmap", heatmaps[0].shape)
    heatmaps = [heatmap.cpu().detach().numpy() for heatmap in heatmaps]
    heatmapStack = np.stack(heatmaps, axis=-1)
    # print("Shape of stack: ", heatmapStack.shape)
    heatmapArgMax = np.argmax(heatmapStack, axis=4)
    # print("Shape of argmax: ", heatmapArgMax.shape)
    # print("Unique values: ", np.unique(heatmapArgMax))
    # print("ARGMAX: ", heatmapArgMax[0])
    # sys.exit()
    # print("Type of Heatmap", type(heatmaps[0]))
    # print("Shape of heatmap", heatmaps[0].shape)
    # sys.exit()
    # masks = [heatmap > threshold for heatmap in heatmaps]
    
    # # Find the overlapping regions among all heatmaps
    # overlap_mask = np.logical_and.reduce(masks)
    
    # # Remove the overlap from the original heatmaps
    # heatmaps_no_overlap = []
    # for heatmap in heatmaps:
    #     heatmap_copy = heatmap.copy()
    #     heatmap_copy[overlap_mask] = 0
    #     heatmap_copy = torch.from_numpy(heatmap_copy)
    #     heatmap_copy = heatmap_copy.cuda()
    #     heatmaps_no_overlap.append(heatmap_copy)
        
    # return heatmaps_no_overlap
    return heatmapArgMax

def getHeatmaps(gradcam, input_image, num_classes):
    AttribsList = []
    for i in range(num_classes):
        # if i == 0 or i == 5:
        # if i ==0:
        #     continue
        # print(type(input_image))
        # print(input_image.shape)
        myAttribs = gradcam.attribute(input_image, i)
        # myAttribs = gradcam.compute_map(x=input_image, class_idx=i)
        # print("Type of myAttribs", type(myAttribs))
        # print("Shape of myAttribs", myAttribs.shape)

        # print("Type of input_image", type(input_image))
        # print("Shape of input_image", input_image.shape)

        myAttribs = _normalize_attr(myAttribs.cpu().detach(), sign="absolute_value")
        # print(type(myAttribs))
        myAttribs = torch.from_numpy(myAttribs)
        myAttribs = myAttribs.cuda()
        # print(myAttribs.shape)
        # if myAttribs.count_nonzero() == 0:
        #     sys.exit()
        AttribsList.append(myAttribs)
        # heatmaps = [heatmap.cpu().detach().numpy() for heatmap in AttribsList]
        # heatmaps = heatmaps[1:]
        toReturn = remove_overlap(AttribsList, threshold=0.5)
        # toReturn = np.stack(heatmaps, axis = -1
        # toReturn = np.argmax(toReturn, axis=-1)
        # print(toReturn.shape)
    
    # if clone:
        
    
    return torch.from_numpy(toReturn)

def convertSegMaps(attribsList, threshold=0.5):
    heatmaps = [heatmap.cpu().detach().numpy() for heatmap in attribsList]
    masks = [heatmap > threshold for heatmap in heatmaps]
    # print(type(masks))


class myDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(myDecoder, self).__init__()
        
        self.deconv1 = nn.ConvTranspose2d(in_channels, 512, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.conv_final = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = F.relu(self.bn5(self.deconv5(x)))
        x = self.conv_final(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ClassificationHead, self).__init__()
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=num_features, out_features=num_classes, bias=True)
        self.swish = monai.networks.blocks.MemoryEfficientSwish()
    
    def forward(self, x):
        x = self.adaptive_avg_pool2d(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.dropout(x)
        x = self.fc(x)
        return x

class EfficientNetClone(pl.LightningModule):
    def __init__(self, num_classes, pretrained=False, ckpt_path="./", base_encoder="efficientnet-b0", lr=1e-4, weight_decay=1e-4):
        # super().__init__(*args, **kwargs)
        super().__init__()
        self.save_hyperparameters()
        self.ckpt_path = ckpt_path
        self.base_encoder = base_encoder
        self.num_classes = num_classes
        self.encoder = EfficientNetBN(in_channels=1, model_name=self.base_encoder, num_classes=self.num_classes, pretrained=pretrained)
        # print(list(self.encoder.children())[:-4])
        # print(self.encoder)
        # myEnc = list(myEnc.children())[:-4]
        # layerList = list(self.encoder.children())[:-4]
        # self.encoder = nn.Sequential(*layerList)
    
        self.feature_maps = None
        self.last_conv_layer = self.encoder._conv_head  # Replace with actual layer
        # print(self.last_conv_layer)
        self.last_conv_layer.register_forward_hook(self._hook_fn)
        
    
    def _hook_fn(self, module, input, output):
        self.feature_maps = output

    
    def forward(self, x):

        fwdPass = self.encoder(x)

        return fwdPass


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer


class EfficientNetEncoder(pl.LightningModule):
    def __init__(self, num_classes, pretrained=False, ckpt_path="./", base_encoder="efficientnet-b0", lr=1e-4, weight_decay=1e-4):
        # super().__init__(*args, **kwargs)
        super().__init__()
        self.save_hyperparameters()
        self.ckpt_path = ckpt_path
        self.base_encoder = base_encoder
        self.num_classes = num_classes
        self.encoder = EfficientNetBN(in_channels=1, model_name=self.base_encoder, num_classes=self.num_classes, pretrained=pretrained)
        # print(list(self.encoder.children())[:-4])
        # print(self.encoder)
        # myEnc = list(myEnc.children())[:-4]
        layerList = list(self.encoder.children())[:-4]
        self.encoder = nn.Sequential(*layerList)
    
        self.feature_maps = None
        self.last_conv_layer = self.encoder[4] #self.encoder._conv_head  # Replace with actual layer
        # print(self.last_conv_layer)
        self.last_conv_layer.register_forward_hook(self._hook_fn)
        
    
    def _hook_fn(self, module, input, output):
        self.feature_maps = output

    
    def forward(self, x):

        fwdPass = self.encoder(x)

        return fwdPass


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer
        
        
class SharedBackboneUNet(pl.LightningModule):
    def __init__(self, num_classes, pretrained=False, ckpt_path="./", base_encoder="efficientnet-b0", lr=1e-3, weight_decay=1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.ckpt_path = ckpt_path
        self.base_encoder = base_encoder
        self.num_classes = num_classes
        self.encoder = EfficientNetEncoder(num_classes = self.num_classes)
        self.classificationHead = ClassificationHead(1280, self.num_classes)
        self.accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=num_classes)
        # self.classificationLoss = FocalLoss(alpha=1, gamma=2)
        self.segmentationLoss = DiceLoss(to_onehot_y=True)
        self.diceWeight = 0.0
        self.gradcamHeatmapsTrain = None
        self.gradcamHeatmapsVal = None
        self.lastXShapeTrain = None
        self.lastXShapeVal = None
        self.alpha = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0])#1.0
        self.gamma = 2.0
        # print(type(self.num_classes))
        self.auroc = torchmetrics.AUROC(num_labels=self.num_classes, average = "weighted", task="multilabel")
        self.decoder = myDecoder(1280, self.num_classes)
        # self.precision_recall = torchmetrics.PrecisionRecallCurve(task="multilabel", num_classes = self.num_classes)
    
    def focal_loss(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss()(inputs, targets) #, reduction='none'
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha.cuda() * (1 - pt)**self.gamma * CE_loss
        return F_loss.mean()

    def forward(self, x):
       
        x = self.encoder(x)

        classification = self.classificationHead(x)

        decoderOut = self.decoder(x)

        return decoderOut, classification
        
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        # print("Training batch x type: ", type(x))
        y = y.float()
        # x = x.unsqueeze(1)
        y_true = torch.where(y.sum(dim=1, keepdim=True) != 0, y / y.sum(dim=1, keepdim=True), torch.zeros_like(y))

        if self.global_step % 5 == 0 or self.global_step == 0 or x.shape != self.lastXShapeTrain:
            # print("Before Model cloned!!!")
            self.clone_model = EfficientNetClone(ckpt_path = self.ckpt_path, num_classes=self.num_classes)
            # print("Before Load Dict!!!")
            self.clone_model.load_state_dict(self.state_dict(), strict=False) #expect the decoder to not be in the clone
            # print("Model cloned!!!")
            self.clone_model.eval()
            self.clone_model = self.clone_model.cuda()
           
            with torch.no_grad():
                # print("SHAPE OF X BEFORE PRED: ", x.shape)
                y_pred_clone = self.clone_model(x.cuda())
            
            myGradcamClone = GuidedGradCam(self.clone_model, self.clone_model.last_conv_layer)
            input_image = x#[:, 0, :, :]
            self.gradcamHeatmapsTrain = getHeatmaps(gradcam=myGradcamClone, input_image=input_image.requires_grad_(True), num_classes=self.num_classes)

        self.gradcamHeatmapsTrain = self.gradcamHeatmapsTrain.cuda()
        outputs_seg, outputs_class = self(x)
        
        loss_classification = self.focal_loss(outputs_class, y_true)  # Replace with your loss function
        loss_segmentation = self.segmentationLoss(outputs_seg, self.gradcamHeatmapsTrain)
        self.lastXShapeTrain = x.shape
        combinedLoss = (1-self.diceWeight) * loss_classification + self.diceWeight * loss_segmentation
        self.log('train_loss', combinedLoss)
        self.log('train_classif', loss_classification)
        self.log('train_seg', loss_segmentation)


        return combinedLoss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        y = y.float()
        y_true = torch.where(y.sum(dim=1, keepdim=True) != 0, y / y.sum(dim=1, keepdim=True), torch.zeros_like(y))
        # Implement your training loop here
        if self.global_step % 5 == 0 or self.global_step == 0 or x.shape != self.lastXShapeVal:
            self.clone_model = EfficientNetClone(ckpt_path = self.ckpt_path, num_classes=self.num_classes, pretrained=False)
            
            self.clone_model.load_state_dict(self.encoder.state_dict(), strict=False)
            self.clone_model.eval()
            self.clone_model = self.clone_model.cuda()
            # self.clone_model = self.clone_model.cuda()
            with torch.no_grad():
                # print("SHAPE OF X BEFORE PRED: ", x.shape)
                y_pred_clone = self.clone_model(x.cuda())
           
            myGradcamClone = GuidedGradCam(self.clone_model, self.clone_model.last_conv_layer)
            
            input_image = x#[:, 0, :, :]
            self.gradcamHeatmapsVal = getHeatmaps(gradcam=myGradcamClone, input_image=input_image.requires_grad_(True), num_classes=self.num_classes)

            # print("Cloned and set to eval mode at step", self.global_step)
            if self.diceWeight < 0.5:
                self.diceWeight = self.diceWeight + 0.05
        # inputs, targets = batch  # Replace with your data loading logic
        self.gradcamHeatmapsVal = self.gradcamHeatmapsVal.cuda()
        outputs_seg, outputs_class = self(x)
        # print(len(outputs_class))
        # print(type(outputs_class))
        loss_classification = self.focal_loss(outputs_class, y_true)  # Replace with your loss function

        loss_segmentation = self.segmentationLoss(outputs_seg, self.gradcamHeatmapsVal)
        self.lastXShapeVal = x.shape

        combinedLoss = (1-self.diceWeight) * loss_classification + self.diceWeight * loss_segmentation
        aucroc_score = self.auroc(outputs_class, y)
        combinedMetric  = aucroc_score - combinedLoss
        # aucpr = multilabel_auprc(outputs_class, y, num_labels=self.num_classes, average=None)
        # precision, recall = self.precision_recall(output_class, y)
        # auprc = auc(recall, precision)
        # self.log("auc_pr", aucpr)
        self.log('combined_metric', combinedMetric)
        self.log('val_auroc', aucroc_score)
        self.log('val_loss', combinedLoss)
        self.log('val_classif', loss_classification)
        self.log('val_seg', loss_segmentation)
    