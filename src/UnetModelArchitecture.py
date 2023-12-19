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
        toReturn = remove_overlap(AttribsList, threshold=0.5)
        # toReturn = np.stack(heatmaps, axis = -1
        # toReturn = np.argmax(toReturn, axis=-1)
        # print(toReturn.shape)
    
    # if clone:
        
    
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
    def __init__(self, num_classes, pretrained=True, ckpt_path="./", base_encoder="efficientnet_b0", lr=1e-3, weight_decay=1e-4):
        # super().__init__(*args, **kwargs)
        super().__init__()
        self.save_hyperparameters()
        self.ckpt_path = ckpt_path
        self.base_encoder = base_encoder
        self.num_classes = num_classes
        self.encoder = EfficientNet(base_encoder="efficientnet-b0").load_from_checkpoint(self.ckpt_path)
        
        self.encoder.convnet.classifier[1] = nn.Linear(self.encoder.convnet.classifier[1].in_features, num_classes)
        self.feature_maps = None
        self.last_conv_layer = self.encoder.convnet.features[8][0]  # Replace with actual layer
        self.last_conv_layer.register_forward_hook(self._hook_fn)
        
    
    def _hook_fn(self, module, input, output):
        self.feature_maps = output

    
    def forward(self, x):
        # fwdPass = self.efficientnet(x)
        fwdPass = self.encoder(x)
        # return torch.sigmoid(fwdPass)
        return fwdPass


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer
        
        
class SharedBackboneUNet(pl.LightningModule):
    def __init__(self, num_classes, pretrained=True, ckpt_path="./", base_encoder="efficientnet_b0", lr=1e-3, weight_decay=1e-4):
        # super(SharedBackboneUNet, self).__init()
        # self.encoder = efficientnet_b0(pretrained=pretrained)
        super().__init__()
        self.save_hyperparameters()
        self.ckpt_path = ckpt_path
        self.base_encoder = base_encoder
        self.num_classes = num_classes
        self.encoder = EfficientNet(base_encoder="efficientnet-b0").load_from_checkpoint(self.ckpt_path)
        self.classificationLoss = FocalLoss(alpha=1, gamma=2)
        self.segmentationLoss = DiceLoss()
        self.diceWeight = 0.0
        self.transition_layer = nn.Conv2d(1280, 320, kernel_size=1)
        
        self.encoder.convnet.classifier[1] = nn.Linear(self.encoder.convnet.classifier[1].in_features, num_classes)
        self.encoder_blocks = [
            self.encoder.convnet.features[0], #first conv_layer
            self.encoder.convnet.features[1],
            self.encoder.convnet.features[2],
            self.encoder.convnet.features[3],
            self.encoder.convnet.features[4],
            self.encoder.convnet.features[5],
            self.encoder.convnet.features[6],
            self.encoder.convnet.features[7],
            self.encoder.convnet.features[8], #last conv_layer
    ]

        # Separate classification head
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, num_classes)  # Adjust the input and output dimensions
        )

        # # Decoder for segmentation
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(320, 32, kernel_size=2, stride=2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, num_classes, kernel_size=1)
        # )
        print(self.encoder_blocks[0])
        print(self.encoder_blocks[8])

        self.decoder_blocks = self.encoder_blocks[::-1]
        print(self.decoder_blocks[0])
        print(self.decoder_blocks[8])

        # Define additional decoder layers and connections
        for i in range(len(self.decoder_blocks)):
            # print("Block: ",i, self.decoder_blocks[i])
            if i == 0 or i == 8:
                in_channels = self.decoder_blocks[i][0].out_channels
                print("In_channels: ", self.decoder_blocks[i][0])
                out_channels = self.decoder_blocks[i][0].in_channels
                print("out_channels: ", self.decoder_blocks[i][0])
            else:    
                print(self.decoder_blocks[i][0].block[0][0].in_channels)
                in_channels = self.decoder_blocks[i][0].block[0][0].out_channels
                print("In_channels: ", self.decoder_blocks[i][0].block[0][0])
                out_channels = self.decoder_blocks[i][0].block[0][0].in_channels  # Assuming you halve the channels in each decoder layer
                print("out_channels: ", self.decoder_blocks[i][0].block[0][0])
            
            decoder_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.decoder_blocks.append(decoder_layer)
            
            # Define an upsampling layer
            upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.decoder_blocks.append(upsample_layer)
        
        # Create the decoder module
        self.decoder = nn.Sequential(*self.decoder_blocks)
        print(self.decoder)
        
        # Final convolution layer to get the desired number of output channels
        self.final_conv = nn.Conv2d(out_channels, num_classes, kernel_size=1)

            # # if i == 3:
            # #     break

            # # Define a decoder layer
            # decoder_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            
            # # Append the decoder layer to your model
            # self.add_module(f'decoder_{i}', decoder_layer)
            # # self.add_module(f'decoder_{i}', self.decoder_blocks[i])
            
            # # Connect the decoder layer to the encoder layer (skip connection)
            # print(in_channels, out_channels)
            # setattr(self, f'upconv_{i}',  nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1))


        # self.decoder = nn.Sequential(
        #     nn.Conv2d(320, 32, kernel_size=1),  # Reduce the number of channels from 320 to 32
        #     nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, num_classes, kernel_size=1)
        # )
        self.eval_encoder = None
        self.eval_classification_head = None

    def forward(self, x):
        skips = []
        for block in self.encoder_blocks:
            # print("SHAPE of X: ", x.shape)
            x = block(x)
            skips.append(x)
        # encoder_output_channels = []
        # for encoder_block in self.encoder_blocks:
        #     if isinstance(encoder_block, nn.Conv2d):
        #         output_channels = encoder_block.out_channels
        #         encoder_output_channels.append(output_channels)
        #     else:
        #         # Handle other layer types if necessary
        #         pass

        # # Print or use the output channels
        # print(encoder_output_channels)

        # Classification head
        classification_output = self.classification_head(x)
        # x = self.transition_layer(x)
        # # Decoder for segmentation
        # print("X-Shape: ", x.shape)
        # x_segmentation = self.decoder(x)
        # # print(x_segmentation.shape)
        # for skip in reversed(skips):
        #     x_segmentation = F.interpolate(x_segmentation, scale_factor=2, mode='bilinear', align_corners=False)
        #     x_segmentation = torch.cat([x_segmentation, skip], dim=1)
        #     x_segmentation = self.decoder(x_segmentation)
        print("SHAPE OF  X: ", x.shape)
        print("SHAPE OF FUSION: ", self.encoder_blocks[8], self.decoder_blocks[0])

        for i in range(0, len(self.decoder_blocks), 2):
            x = self.decoder[i](x)
            encoder_output = skips.pop()
            x = torch.cat((x, encoder_output), dim=1)
            x = self.decoder[i + 1](x)
        
        # Final convolution
        x = self.final_conv(x)

        return x, classification_output
    """
    # for i, decoder_block in enumerate(self.decoder_blocks):
        #     print("Shape of Decoder block: ", decoder_block)
        #     x = getattr(self, f'upconv_{i}')(x)  # Upsample using 1x1 convolution
        #     encoder_output = skips.pop()  # Get the corresponding encoder output
        #     print("SHAPE OF encoder_output: ", encoder_output.shape)
        #     x = torch.cat((x, encoder_output), dim=1)  # Concatenate with the encoder output
        #     print("SHAPE OF  X_cat: ", x.shape)
            
        #     x = decoder_block(x)

        # # Use a suitable activation function for multilabel classification
        # x = torch.sigmoid(x)  # Sigmoid activation for each channel?? TODO 
        # print(x.shape) #TODO FIND DIMENSIONS OF SEGMENTATION
    """

        
        
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        # Implement your training loop here
        gradcam_heatmaps = None
        if self.global_step % 5 == 0 or self.global_step == 0:
            self.clone_model = EfficientNetClone(ckpt_path = self.ckpt_path, num_classes=self.num_classes)
            self.clone_model.load_state_dict(self.state_dict(), strict=False) #expect the decoder to not be in the clone
            self.clone_model.eval()
            self.clone_model = self.clone_model.cuda()
            with torch.no_grad():
                y_pred_clone = self.clone_model(x.cuda())
            myGradcamClone = GuidedGradCam(self.clone_model, self.clone_model.last_conv_layer)
            gradcam_heatmaps = getHeatmaps(gradcam=myGradcamClone, input_image=x.requires_grad_(True), num_classes=self.num_classes)
            # self.eval_encoder = self.encoder.clone()
            # self.eval_classification_head = self.classification_head.clone()
            # self.eval_encoder.eval()
            # self.eval_classification_head.eval()

            print("Cloned and set to eval mode at step", self.global_step)
        # inputs, targets = batch  # Replace with your data loading logic
        outputs_seg, outputs_class = self(x)
        loss_classification = self.classificationLoss(outputs_class, y)  # Replace with your loss function
        loss_segmentation = self.sementationLoss(outputs_seg, gradcam_heatmaps)

        combinedLoss = (1-self.diceWeight) * loss_classification + self.diceWeight * loss_segmentation
        self.log('train_loss', combinedLoss)
        self.log('train_classif', loss_classification)
        self.log('train_seg', loss_segmentation)


        return combinedLoss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Implement your training loop here
        if self.global_step % 5 == 0 or self.global_step == 0:
            self.clone_model = EfficientNetClone(ckpt_path = self.ckpt_path, num_classes=self.num_classes)
            self.clone_model.load_state_dict(self.state_dict(), strict=False)
            self.clone_model.eval()
            self.clone_model = self.clone_model.cuda()
            with torch.no_grad():
                y_pred_clone = self.clone_model(x.cuda())
            myGradcamClone = GuidedGradCam(self.clone_model, self.clone_model.last_conv_layer)
            gradcam_heatmaps = getHeatmaps(gradcam=myGradcamClone, input_image=x.requires_grad_(True), num_classes=self.num_classes)
            print("GRADCAM HEATMAP SHAPE: ",gradcam_heatmaps.shape)
            # self.eval_encoder = self.encoder.clone()
            # self.eval_classification_head = self.classification_head.clone()
            # self.eval_encoder.eval()
            # self.eval_classification_head.eval()

            print("Cloned and set to eval mode at step", self.global_step)
        # inputs, targets = batch  # Replace with your data loading logic
        outputs_seg, outputs_class = self(x)
        loss_classification = self.classificationLoss(outputs_class, y)  # Replace with your loss function
        loss_segmentation = self.sementationLoss(outputs_seg, gradcam_heatmaps)

        combinedLoss = (1-self.diceWeight) * loss_classification + self.diceWeight * loss_segmentation
        self.log('val_loss', combinedLoss)
        self.log('val_classif', loss_classification)
        self.log('val_seg', loss_segmentation)
    