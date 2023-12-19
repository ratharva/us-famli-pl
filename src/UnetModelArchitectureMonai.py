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
from monai.networks.nets import FlexibleUNet, EfficientNetEncoder, EfficientNetBN
from monai.visualize.class_activation_maps import GradCAMpp
import monai.visualize.class_activation_maps as cam
import monai
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
        F_loss = self.alpha * (1 - pt)**self.gamma * CE_loss
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

# def colorize_segmentation_map(seg_map, colormap):
#     colored_map = np.zeros((3, seg_map.shape[1], seg_map.shape[2]), dtype=np.uint8)
    
#     for label, color in colormap.items():
#         mask = (seg_map == label)
#         for i in range(5):  # 3 channels for RGB
#             colored_map[i, mask] = color[i]
    
#     return colored_map

# def colorize_segmentation_map(seg_map, colormap):
#     colored_map = np.zeros((3, seg_map.shape[0], seg_map.shape[1], seg_map.shape[2]), dtype=np.uint8)
    
#     for label, color in colormap.items():
#         mask = (seg_map == label)
#         for i in range(3):  # 3 channels for RGB
#             colored_map[i, mask] = color[i]
    
#     return colored_map

def colorize_segmentation_map(seg_map, colormap):
    batch_size, num_classes, height, width = seg_map.shape
    colored_maps = np.zeros((batch_size, 3, height, width), dtype=np.uint8)
    
    for label, color in colormap.items():
        mask = (seg_map == label)
        for i in range(3):  # 3 channels for RGB
            colored_maps[:, i, :, :][mask] = color[i]
    
    return colored_maps


class EfficientNetClone(pl.LightningModule):
    def __init__(self, num_classes, pretrained=False, ckpt_path="./", base_encoder="efficientnet-b0", lr=1e-3, weight_decay=1e-4):
        # super().__init__(*args, **kwargs)
        super().__init__()
        self.save_hyperparameters()
        self.ckpt_path = ckpt_path
        self.base_encoder = base_encoder
        self.num_classes = num_classes
        # self.encoder = EfficientNetEncoder(base_encoder="efficientnet-b0", num_classes=self.num_classes).load_from_checkpoint(self.ckpt_path)
        # self.encoder = EfficientNetEncoder(model_name=base_encoder, num_classes=self.num_classes, pretrained=True)
        self.encoder = EfficientNetBN(in_channels=1, model_name=base_encoder, num_classes=self.num_classes, pretrained=pretrained)
        
        
        # self.encoder.convnet.classifier[1] = nn.Linear(self.encoder.convnet.classifier[1].in_features, num_classes)
        self.feature_maps = None
        self.last_conv_layer = self.encoder._conv_head  # Replace with actual layer
        # print(self.last_conv_layer)
        self.last_conv_layer.register_forward_hook(self._hook_fn)
        
    
    def _hook_fn(self, module, input, output):
        self.feature_maps = output

    
    def forward(self, x):
        # fwdPass = self.efficientnet(x)
        # print("SHAPE OF X: ", x.shape)
        fwdPass = self.encoder(x)
        # return torch.sigmoid(fwdPass)
        return fwdPass


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer
        
        
class SharedBackboneUNet(pl.LightningModule):
    def __init__(self, num_classes, pretrained=False, ckpt_path="./", base_encoder="efficientnet-b0", lr=1e-3, weight_decay=1e-4):
        # super(SharedBackboneUNet, self).__init()
        # self.encoder = efficientnet_b0(pretrained=pretrained)
        super().__init__()
        self.save_hyperparameters()
        self.ckpt_path = ckpt_path
        self.base_encoder = base_encoder
        self.num_classes = num_classes
        self.model = FlexibleUNet(in_channels=1, out_channels=self.num_classes, pretrained=False, backbone=self.base_encoder, decoder_channels=(1280, 320, 192, 112, 80, 40, 24, 16, 32))#(256, 128, 64, 32, 16)) , 24, 16, 32
        # print(len(self.model.encoder.encoder_channels))
        # print("Encoder Channel Length: ", len(self.model.decoder.encoder_channels))
        self.accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=num_classes)
        self.model.encoder._fc = nn.Linear(1280, self.num_classes)
        self.classificationLoss = FocalLoss(alpha=1, gamma=2)
        self.segmentationLoss = DiceLoss(to_onehot_y=True)
        self.diceWeight = 0.0
        self.gradcamHeatmapsTrain = None
        self.gradcamHeatmapsVal = None
        self.lastXShapeTrain = None
        self.lastXShapeVal = None
        # print(type(self.num_classes))
        self.auroc = torchmetrics.AUROC(num_labels=self.num_classes, average='macro', task="multilabel")
   

    def forward(self, x):
        # x_copy = copy.deepcopy(x)
        # print(self.model.encoder)
        # x = self.model.encoder._conv_stem(x)
        # x = self.model.encoder._conv_stem_padding(x)
        # x = self.model.encoder._bn0(x)
        # x = self.model.encoder._blocks[0](x)
        # x = self.model.encoder._blocks[1](x)
        # x = self.model.encoder._blocks[2](x)
        # x = self.model.encoder._blocks[3](x)
        # x = self.model.encoder._blocks[4](x)
        # x = self.model.encoder._blocks[5](x)
        # print(x.shape)
        # x = self.model.encoder._blocks[6](x)
        # print(x.shape)
        # x = self.model.encoder._conv_head(x)
        # x = self.model.encoder._conv_head_padding(x)
        # x = self.model.encoder._bn1(x)
        # x = self.model.encoder._avg_pooling(x)
        # x = x.view(x.size(0), -1)
        # x = self.model.encoder._dropout(x)
        # print(x.shape)
        # x = self.model.encoder._fc(x)

        encoderOut = self.model.encoder(x)
        # for i in range(len(encoderOut)):
        #     print("Classification Output: ",i, classification[i].shape)
        # Apply Classification Head
        classification = self.model.encoder._conv_head(encoderOut[-1])
        classification = self.model.encoder._conv_head_padding(classification)
        classification = self.model.encoder._bn1(classification)
        classification = self.model.encoder._avg_pooling(classification)
        classification = classification.view(classification.size(0), -1)
        classification = self.model.encoder._dropout(classification)
        classification = self.model.encoder._fc(classification)
        # print("Classification shape: ", classification.shape)
        #classification shape (32, 16, 128, 128)
        #pass classification[n] through classification head

        

        decoderOut = self.model.decoder(encoderOut, self.model.skip_connect)
        fullModel = self.model.segmentation_head(decoderOut)
        # print(fullModel[0].shape)
        # print(torch.eq(classification[0], fullModel[0]))

        return fullModel, classification
        
    
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
        # Implement your training loop here
        # gradcam_heatmaps = None
        if self.global_step % 5 == 0 or self.global_step == 0 or x.shape != self.lastXShapeTrain:
            # print("Before Model cloned!!!")
            self.clone_model = EfficientNetClone(ckpt_path = self.ckpt_path, num_classes=self.num_classes)
            # print("Before Load Dict!!!")
            self.clone_model.load_state_dict(self.state_dict(), strict=False) #expect the decoder to not be in the clone
            # print("Model cloned!!!")
            self.clone_model.eval()
            self.clone_model = self.clone_model.cuda()
            # for p1, p2 in zip(self.parameters(), self.clone_model.parameters()):
            #     if not torch.equal(p1, p2):
            #         print("Different parameters found!")
            #         break
            with torch.no_grad():
                # print("SHAPE OF X BEFORE PRED: ", x.shape)
                y_pred_clone = self.clone_model(x.cuda())
            # print("LAST LAYER: ", self.clone_model)
            # myGradcamClone = GradCAMpp(self.clone_model, target_layers="encoder._conv_head")
            myGradcamClone = GuidedGradCam(self.clone_model, self.clone_model.last_conv_layer)
            input_image = x#[:, 0, :, :]
            self.gradcamHeatmapsTrain = getHeatmaps(gradcam=myGradcamClone, input_image=input_image.requires_grad_(True), num_classes=self.num_classes)
            # print(self.gradcamHeatmapsTrain)
            # self.eval_encoder = self.encoder.clone()
            # self.eval_classification_head = self.classification_head.clone()
            # self.eval_encoder.eval()
            # self.eval_classification_head.eval()

            # print("Cloned and set to eval mode at step", self.global_step)
        # inputs, targets = batch  # Replace with your data loading logic
        # print(gradcam_heatmaps)
        self.gradcamHeatmapsTrain = self.gradcamHeatmapsTrain.cuda()
        outputs_seg, outputs_class = self(x)
        
        loss_classification = self.classificationLoss(outputs_class, y_true)  # Replace with your loss function
        loss_segmentation = self.segmentationLoss(outputs_seg, self.gradcamHeatmapsTrain)
        self.lastXShapeTrain = x.shape
        combinedLoss = (1-self.diceWeight) * loss_classification + self.diceWeight * loss_segmentation
        self.log('train_loss', combinedLoss)
        self.log('train_classif', loss_classification)
        self.log('train_seg', loss_segmentation)


        return combinedLoss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # print("Validation batch x type: ", type(x))
        # x = x.unsqueeze(1)
        y = y.float()
        y_true = torch.where(y.sum(dim=1, keepdim=True) != 0, y / y.sum(dim=1, keepdim=True), torch.zeros_like(y))
        # Implement your training loop here
        if self.global_step % 5 == 0 or self.global_step == 0 or x.shape != self.lastXShapeVal:
            self.clone_model = EfficientNetClone(ckpt_path = self.ckpt_path, num_classes=self.num_classes, pretrained=False)
            
            self.clone_model.load_state_dict(self.model.encoder.state_dict(), strict=False)
            # print("Model cloned!!!")
            # print("CLONE MODEL: ", self.clone_model)
            # for p1, p2 in zip(self.parameters(), self.clone_model.parameters()):
            #     if not torch.equal(p1, p2):
            #         print("Different parameters found!", p1, p2)
            #         break
            # different_params = []
            # for (name1, param1), (_, param2) in zip(self.clone_model.encoder.named_parameters(), self.clone_model.named_parameters()):
            #     if not torch.equal(param1, param2):
            #         different_params.append(name1)

            # if different_params:
            #     print("Different parameters found in layers:")
            #     for param_name in different_params:
            #         print(param_name)
            # else:
            #     print("No differences found!")
            self.clone_model.eval()
            self.clone_model = self.clone_model.cuda()
            # self.clone_model = self.clone_model.cuda()
            with torch.no_grad():
                # print("SHAPE OF X BEFORE PRED: ", x.shape)
                y_pred_clone = self.clone_model(x.cuda())
            # print("LAST LAYER: ", self.clone_model)
            # for name, _ in self.clone_model.named_modules(): print(name)
            # myGradcamClone = cam.GradCAMpp(nn_module=self.clone_model, target_layers="encoder._conv_head")
            myGradcamClone = GuidedGradCam(self.clone_model, self.clone_model.last_conv_layer)
            # myHook = cam.ModelWithHooks(self.clone_model, target_layer_names="encoder._conv_head")
            # myGradcamClone = cam.GradCAMpp(myHook, "encoder._conv_head")
            # x = x.requires_grad_(True)
            # myRes = myGradcamClone.compute_map(x=x, class_idx=2, retain_graph=True)
            input_image = x#[:, 0, :, :]
            self.gradcamHeatmapsVal = getHeatmaps(gradcam=myGradcamClone, input_image=input_image.requires_grad_(True), num_classes=self.num_classes)
            # print("GRADCAM HEATMAP SHAPE: ",gradcam_heatmaps.shape)
            # self.eval_encoder = self.encoder.clone()
            # self.eval_classification_head = self.classification_head.clone()
            # self.eval_encoder.eval()
            # self.eval_classification_head.eval()

            # print("Cloned and set to eval mode at step", self.global_step)
            if self.diceWeight < 0.5:
                self.diceWeight = self.diceWeight + 0.05
        # inputs, targets = batch  # Replace with your data loading logic
        self.gradcamHeatmapsVal = self.gradcamHeatmapsVal.cuda()
        outputs_seg, outputs_class = self(x)
        # print(len(outputs_class))
        # print(type(outputs_class))
        loss_classification = self.classificationLoss(outputs_class, y_true)  # Replace with your loss function
        # print(outputs_seg.device, gradcam_heatmaps.device)
        # print("Input: ", type(outputs_seg), "GroundTruth: ", type(self.gradcamHeatmapsVal))
        # print(outputs_seg.shape, self.gradcamHeatmapsVal.shape)
        loss_segmentation = self.segmentationLoss(outputs_seg, self.gradcamHeatmapsVal)
        self.lastXShapeVal = x.shape

        combinedLoss = (1-self.diceWeight) * loss_classification + self.diceWeight * loss_segmentation
        aucroc_score = self.auroc(outputs_class, y)
        combinedMetric  = aucroc_score - combinedLoss
        self.log('combined_metric', combinedMetric)
        self.log('val_auroc', aucroc_score)
        self.log('val_loss', combinedLoss)
        self.log('val_classif', loss_classification)
        self.log('val_seg', loss_segmentation)
    