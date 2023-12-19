# from monai.losses import DiceBCELoss
from typing import List, Union
from monai.losses import DiceLoss
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
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
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
import cv2

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

def dice_loss(heatmap1, heatmap2):
    numerator = 2 * np.sum(heatmap1 * heatmap2)
    denominator = np.sum(heatmap1) + np.sum(heatmap2)
    return 1 - numerator / (denominator + 1e-6)

def remove_overlap(heatmaps, threshold=0.5, sigma = 0.1, dilation_size=3, erosion_size=3):
    # Create binary masks based on the threshold
    heatmaps = [heatmap.cpu().detach().numpy() for heatmap in heatmaps]
    masks = [heatmap > threshold for heatmap in heatmaps]

    # smoothed_heatmaps = [gaussian_filter(heatmap, sigma=sigma) for heatmap in heatmaps]
    
    # Find the overlapping regions among all heatmaps
    overlap_mask = np.logical_and.reduce(masks)
    # print("TYPE OF OVERLAP MASK IS: ", type(overlap_mask))
    # print("Value counts: ", np.unique(overlap_mask))

    # # Apply dilation to smoothen the overlap regions
    # dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    # dilated_mask = cv2.dilate(overlap_mask.astype(np.uint8), dilation_kernel, iterations=1)
    
    # # Apply erosion to refine the boundaries
    # erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    # eroded_mask = cv2.erode(dilated_mask, erosion_kernel, iterations=1)
    # if np.unique(overlap_mask) != [False]:
    #     dilation_structure = np.ones((dilation_size, dilation_size))
    #     overlap_mask = binary_dilation(overlap_mask, structure=dilation_structure)
        
    #     # Perform erosion to smooth the edges of the preserved regions
    #     erosion_structure = np.ones((erosion_size, erosion_size))
    #     overlap_mask = binary_erosion(overlap_mask, structure=erosion_structure)
    
    
    # Remove the overlap from the original heatmaps
    heatmaps_no_overlap = []
    for heatmap in heatmaps:
        heatmap_copy = heatmap.copy()
        heatmap_copy[overlap_mask] = 0
        heatmap_copy = torch.from_numpy(heatmap_copy)
        heatmap_copy = heatmap_copy.cuda()
        heatmaps_no_overlap.append(heatmap_copy)
        
    return heatmaps_no_overlap

def getHeatmaps(gradcam, input_image, num_classes, clone):
    AttribsList = []
    for i in range(num_classes):
        if i == 0 or i == 5:
            continue
        myAttribs = gradcam.attribute(input_image, i)
        AttribsList.append(myAttribs)
    
    if not clone:
        AttribsList = remove_overlap(AttribsList, threshold=0.5)
    
    return AttribsList
        

class EfficientNetClone(pl.LightningModule):
    def __init__(self, args = None, base_encoder='efficientnet_b0', num_classes=6, lr=1e-3, weight_decay=1e-4, class_weights=None, ckpt_path = "./"):
        super().__init__()
        self.save_hyperparameters()
        self.ckpt_path = ckpt_path
        self.base_encoder = base_encoder
        self.num_classes = num_classes
        self.efficientnet = EfficientNet(base_encoder="efficientnet-b0").load_from_checkpoint(self.ckpt_path)
        
        self.efficientnet.convnet.classifier[1] = nn.Linear(self.efficientnet.convnet.classifier[1].in_features, num_classes)

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)
        self.myLoss = nn.CrossEntropyLoss(weight=class_weights)
        # self.myLoss = FocalLoss(alpha=1, gamma=2)
        # self.myLoss = nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=num_classes)
        # print(dir(self.efficientnet))
        # print(self.convnet)
        self.feature_maps = None
        self.last_conv_layer = self.efficientnet.convnet.features[8][0]  # Replace with actual layer
        self.last_conv_layer.register_forward_hook(self._hook_fn)
        self.dice_weight = 0.0
        self.val_dice = None
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
            # gradcam_heatmaps_clone = get_gradcam_heatmap(self.clone_model, self.clone_model.feature_maps.detach(), x.requires_grad_(True), clone=True)
            diceLoss = DiceLoss()
            dice_losses = [diceLoss(h1, h2) for h1, h2 in zip(gradcam_heatmaps, gradcam_heatmaps_clone)]
            avg_dice_loss = torch.mean(torch.stack(dice_losses))

        
        combined_loss = (1 - self.dice_weight) * loss + self.dice_weight * avg_dice_loss
        # combined_loss = loss
        self.combinedLoss = combined_loss
        self.avg_dice = avg_dice_loss
        self.BCELoss = loss
        self.log('train_loss', combined_loss)
        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)
        return combined_loss

    def training_epoch_end(self, outputs):
        # Clone the model and set it to eval mode after each epoch
        self.clone_model = EfficientNetClone(ckpt_path = self.ckpt_path, num_classes=self.num_classes)
        self.clone_model.load_state_dict(self.state_dict(), strict=False)
        self.clone_model.eval()
        
        # if(self.current_epoch + 1) > 4:
        #     self.dice_weight = 0.05
        # self.dice_weight = self.dice_weight + 0.005
        # if(self.current_epoch + 1) > 30:
        if (self.current_epoch + 1) % 5 == 0 and self.dice_weight <=0.5:
            self.dice_weight = self.dice_weight + 0.05
            print("Dice loss weight: ", self.dice_weight)
            print("Dice loss: ", self.avg_dice)
            print("BCE: ", self.BCELoss)
            

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x_copy = copy.deepcopy(x)
        y = y.float()
        y_true = torch.where(y.sum(dim=1, keepdim=True) != 0, y / y.sum(dim=1, keepdim=True), torch.zeros_like(y))
        x = self(x)
        myLoss = self.myLoss(x, y_true)
        myGradcam = GuidedGradCam(self, self.last_conv_layer)
        self.gradcam_heatmaps = getHeatmaps(gradcam=myGradcam, input_image=x_copy.requires_grad_(True), num_classes=self.num_classes, clone=False)
        # loss = nn.BCEWithLogitsLoss()(x, y)
        if self.val_dice == None:
            loss = myLoss
        else:
            print("Entered else!")
            diceLoss = DiceLoss()
            dice_losses = [diceLoss(h1, h2) for h1, h2 in zip(self.gradcam_heatmaps, self.last_epoch_heatmaps)]
            self.val_dice = torch.mean(torch.stack(dice_losses))
            loss = (1 - self.dice_weight) * myLoss + self.dice_weight * self.val_dice
        
        self.log('val_loss', loss, sync_dist=True)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy, sync_dist=True)
        # print(loss)
    
    # def validation_epoch_end(self, outputs):
    #     if self.current_epoch > 3:
    #         self.last_epoch_heatmaps = self.gradcam_heatmaps