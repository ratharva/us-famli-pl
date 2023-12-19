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

def getHeatmaps(gradcam, input_image, num_classes, clone):
    AttribsList = []
    for i in range(num_classes):
        # if i == 0 or i == 5:
        # if i ==0:
        #     continue
        myAttribs = gradcam.attribute(input_image, i)
        # print("Type of myAttribs", type(myAttribs))
        # print("Shape of myAttribs", myAttribs.shape)

        # print("Type of input_image", type(input_image))
        # print("Shape of input_image", input_image.shape)

        myAttribs = _normalize_attr(myAttribs, sign="absolute_value")
        myAttribs = torch.from_numpy(myAttribs)
        myAttribs = myAttribs.cuda()
        # print(myAttribs.shape)
        # if myAttribs.count_nonzero() == 0:
        #     sys.exit()
        AttribsList.append(myAttribs)
        heatmaps = [heatmap.cpu().detach().numpy() for heatmap in AttribsList]
        # heatmaps = heatmaps[1:]
    toReturn = np.hstack(heatmaps)
        # toReturn = np.stack(heatmaps, axis = -1
        # toReturn = np.argmax(toReturn, axis=-1)
        # print(toReturn.shape)
    
    if clone:
        toReturn = remove_overlap(AttribsList, threshold=0.5)
    
    return torch.from_numpy(toReturn)

def convertSegMaps(attribsList, threshold=0.5):
    heatmaps = [heatmap.cpu().detach().numpy() for heatmap in attribsList]
    masks = [heatmap > threshold for heatmap in heatmaps]
    print(type(masks))

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
    def __init__(self, args = None, base_encoder='efficientnet_b0', num_classes=6, lr=1e-3, weight_decay=1e-4, class_weights=None): #, ckpt_path = "./"
        super().__init__()
        self.save_hyperparameters()
        # self.ckpt_path = ckpt_path
        self.base_encoder = base_encoder
        self.num_classes = num_classes
        self.efficientnet = EfficientNet(base_encoder="efficientnet-b0")#.load_from_checkpoint(self.ckpt_path, strict=False)
        
        self.efficientnet.convnet.classifier[1] = nn.Linear(self.efficientnet.convnet.classifier[1].in_features, num_classes)

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)
        # self.myLoss = nn.CrossEntropyLoss(weight=class_weights)
        self.myLoss = FocalLoss(alpha=1, gamma=2)
        # self.myLoss = nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=num_classes)
        # print(dir(self.efficientnet))
        # print(self.convnet)
        self.feature_maps = None
        self.last_conv_layer = self.efficientnet.convnet.features[8][0]  # Replace with actual layer
        self.last_conv_layer.register_forward_hook(self._hook_fn)
        self.dice_weight = 0.0
        self.colormap = {
            0: [0, 0, 0], #class black
            1: [255, 0, 0],   # Class 0 in red
            2: [0, 255, 0],   # Class 1 in green
            3: [0, 0, 255],    # Class 2 in blue
            4: [255, 255, 0]
            # Add more class-color mappings as needed
        }
        # RuntimeError()
    
    def _hook_fn(self, module, input, output):
        self.feature_maps = output

    
    def forward(self, x):
        # fwdPass = self.efficientnet(x)
        fwdPass = self.efficientnet(x)
        # return torch.sigmoid(fwdPass)
        return fwdPass


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.float()
        y_true = torch.where(y.sum(dim=1, keepdim=True) != 0, y / y.sum(dim=1, keepdim=True), torch.zeros_like(y))
        x_copy = copy.deepcopy(x)
        x = self(x)
        loss = self.myLoss(x, y_true)
        
        
        
        myGradcam = GuidedGradCam(self, self.last_conv_layer)
        gradcam_heatmaps = getHeatmaps(gradcam=myGradcam, input_image=x_copy.requires_grad_(True), num_classes=self.num_classes, clone=False)
        seg_map_np = gradcam_heatmaps.cpu().numpy()
        colored_seg_maps = colorize_segmentation_map(seg_map_np, self.colormap)
        self.logger.experiment.add_image('Training_heatmap_' + str(self.current_epoch), colored_seg_maps, self.current_epoch, dataformats='NCHW')
        # print("The image is: ", gradcam_heatmaps[0])
        # print(gradcam_heatmaps.shape)
        # for i in range(len(gradcam_heatmaps)):
        #     tempHeatmap = gradcam_heatmaps[i]
        #     # tempHeatmap = tempHeatmap.cpu()
        #     # tempHeatmap = np.transpose(tempHeatmap, (0, 2, 3, 1))
        #     self.logger.experiment.add_image('Training_heatmap_' + str(self.current_epoch) + "_" + str(i), tempHeatmap, self.current_epoch, dataformats='NCHW')
        
        # gradcam_heatmaps = get_gradcam_heatmap(self, self.feature_maps.detach(), x.requires_grad_(True), clone=False)
        avg_dice_loss = torch.tensor(0.0, device=self.device)

        if hasattr(self, 'clone_model'):
            # print("Entered has_attr")
            # with torch.no_grad():
            self.clone_model = self.clone_model.cuda()
            # print(x.shape)
            with torch.no_grad():
                y_pred_clone = self.clone_model(x_copy.cuda())
            myGradcamClone = GuidedGradCam(self.clone_model, self.clone_model.last_conv_layer)
            gradcam_heatmaps_clone = getHeatmaps(gradcam=myGradcamClone, input_image=x_copy.requires_grad_(True), num_classes=self.num_classes, clone=True)
            seg_map_clone_np = gradcam_heatmaps_clone.cpu().numpy()
            colored_seg_maps_clone = colorize_segmentation_map(seg_map_clone_np, self.colormap)
            self.logger.experiment.add_image('Eval_heatmap_' + str(self.current_epoch), colored_seg_maps_clone, self.current_epoch, dataformats='NCHW') #np.argmax(colored_seg_maps_clone, axis=1).unsqueeze(1)
            # top_100_values = np.partition(gradcam_heatmaps_clone[0], -10)[-10:]
            # # Sort the top 100 values in descending order (optional)
            # top_100_values_sorted = np.sort(top_100_values)[::-1]
            # print("The image is: ", top_100_values_sorted)
            # for i in range(len(gradcam_heatmaps_clone)):
            #     tempHeatmap = gradcam_heatmaps_clone[i]
            #     # tempHeatmap = tempHeatmap.cpu()
            #     # tempHeatmap = np.transpose(tempHeatmap, (0, 2, 3, 1))
            #     self.logger.experiment.add_image('Eval_heatmap_' + str(self.current_epoch) + "_" + str(i), tempHeatmap, self.current_epoch, dataformats='NCHW')
            
            # gradcam_heatmaps_clone = get_gradcam_heatmap(self.clone_model, self.clone_model.feature_maps.detach(), x.requires_grad_(True), clone=True)
            diceLoss = DiceLoss(include_background=False, to_onehot_y=True)
            # trainingArray = np.concatenate(gradcam_heatmaps,axis=2)
            # evalArray = np.stack(gradcam_heatmaps_clone, axis=0)
            # avg_dice_loss = dice_loss(trainingArray, evalArray)
            # print(type(gradcam_heatmaps[0]))
            # print(type(gradcam_heatmaps_clone[0]))
            # gradcam_heatmaps_clone = torch.tensor(gradcam_heatmaps_clone)
            # dice_losses = [diceLoss(h1, h2) for h1, h2 in zip(gradcam_heatmaps, gradcam_heatmaps_clone)]
            # avg_dice_loss = torch.mean(torch.stack(dice_losses))
            avg_dice_loss = diceLoss(gradcam_heatmaps_clone, gradcam_heatmaps)
            self.log_dict({'avg_dice_loss': avg_dice_loss, "dice_weight": self.dice_weight})

        
        combined_loss = (1 - self.dice_weight) * loss + self.dice_weight * avg_dice_loss
        # combined_loss = loss
        self.combinedLoss = combined_loss
        self.avg_dice = avg_dice_loss
        self.BCELoss = loss
        self.log('train_loss', combined_loss)
        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)
        return combined_loss
    

    # def training_step_end(self, outputs):
    #     # Clone the model and set it to eval mode after each epoch
    #     self.clone_model = EfficientNetClone(ckpt_path = self.ckpt_path, num_classes=self.num_classes)
    #     self.clone_model.load_state_dict(self.state_dict(), strict=False)
    #     self.clone_model.eval()
    #     # if(self.current_epoch + 1) > 30:
    #     print("Current training step is: ", self.training_step)
    #     if (self.global_step + 1) % 5 == 0 and self.dice_weight <=0.5:
    #         self.dice_weight = self.dice_weight + 0.005
    #         print("Dice loss weight: ", self.dice_weight)
    #         print("Dice loss: ", self.avg_dice)
    #         print("BCE: ", self.BCELoss)
    
    def training_epoch_end(self, outputs):
        # Clone the model and set it to eval mode after each epoch
        self.clone_model = EfficientNetClone(num_classes=self.num_classes) #ckpt_path = self.ckpt_path,
        self.clone_model.load_state_dict(self.state_dict(), strict=False)
        self.clone_model.eval()
        # if(self.current_epoch + 1) > 30:
        # print("Current training step is: ", self.training_step)
        # if (self.current_epoch + 1) % 5 == 0 and self.dice_weight <=0.5:
        self.dice_weight = self.dice_weight + 0.005
        # print("Dice loss weight: ", self.dice_weight)
        # print("Dice loss: ", self.avg_dice)
        # print("BCE: ", self.BCELoss)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = y.float()
        y_true = torch.where(y.sum(dim=1, keepdim=True) != 0, y / y.sum(dim=1, keepdim=True), torch.zeros_like(y))
        x = self(x)
        
        # loss = nn.BCEWithLogitsLoss()(x, y)
        loss = self.myLoss(x, y_true)
        
        self.log('val_loss', loss, sync_dist=True)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy, sync_dist=True)
    
    # def validation_epoch_end(self, outputs):
        
    