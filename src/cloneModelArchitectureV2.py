# from monai.losses import DiceBCELoss
from monai.losses import DiceLoss
import pytorch_lightning as pl
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

def get_gradcam_heatmap(model, feature_maps, input_image, clone):
    # print("HERE1")
    feature_maps = feature_maps.requires_grad_(True)
    print(feature_maps.shape)
    heatmaps = []
    overlap_removed_heatmaps = []
    overlap_mask = None
    
    # Function to capture gradients
    grads = []
    def save_grad(grad):
        grads.append(grad)
    # print("HERE2")
    # Register hook to capture the gradients
    h = feature_maps.register_hook(save_grad)
    # print("HERE3")
    for i in range(feature_maps.shape[1]):
        # print("HERE4:", i)
        model.zero_grad()
        feature_map_avg = feature_maps[:, i, :, :].mean()
        # print("HERE5:", i)
        # Backpropagation to get the gradients
        feature_map_avg.backward(retain_graph=True)
        
        pooled_grads = torch.mean(grads[-1], dim=[0, 2, 3])
        feature_maps_with_grads = feature_maps.clone()
        # print("HERE6:", i)
        for j in range(feature_maps.shape[1]):
            feature_maps_with_grads[:, j, :, :] *= pooled_grads[j]
        # print("HERE7:", i)
        heatmap = torch.mean(feature_maps_with_grads, dim=1).squeeze()
        heatmap = nn.ReLU()(heatmap)
        heatmap /= torch.max(heatmap)
        
        heatmaps.append(heatmap.detach().cpu().numpy())
        
        if not clone:
            binary_heatmap = (heatmap > 0.2).float()  # Threshold can be changed
            if overlap_mask is None:
                overlap_mask = binary_heatmap
            else:
                overlap_mask *= binary_heatmap
                
            overlap_removed_heatmap = heatmap * (1 - overlap_mask)
            overlap_removed_heatmaps.append(overlap_removed_heatmap.detach().cpu().numpy())
    
    # Remove the hook
    h.remove()
    
    return overlap_removed_heatmaps if not clone else heatmaps


def dice_loss(heatmap1, heatmap2):
    numerator = 2 * np.sum(heatmap1 * heatmap2)
    denominator = np.sum(heatmap1) + np.sum(heatmap2)
    return 1 - numerator / (denominator + 1e-6)

def remove_overlap(heatmaps, threshold=0.5):
    # Create binary masks based on the threshold
    heatmaps = [heatmap.cpu().detach().numpy() for heatmap in heatmaps]
    masks = [heatmap > threshold for heatmap in heatmaps]
    
    # Find the overlapping regions among all heatmaps
    overlap_mask = np.logical_and.reduce(masks)
    
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
        # self.efficientnet = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
        self.efficientnet = timm.create_model("efficientnetv2_rw_m", pretrained=True, num_classes=self.num_classes)
        
        # print(self.efficientnet)
        # self.efficientnet = nn.Sequential(*(list(self.efficientnet.children())[:-1]), nn.Linear(2152, num_classes))  # Adjust the output size to fit the number of classes
        self.sigmoid = nn.Sigmoid()
        # print(self.efficientnet)
        # print(self.efficientnet)
        # self.efficientnet = nn.Sequential(*(list(self.efficientnet.children())[:-1]), nn.Linear(2152, num_classes))  # Adjust the output size to fit the number of classes
        # self.classifier = nn.Linear(1280, num_classes)
        self.sigmoid = nn.Sigmoid()
        # self.myLoss = DiceBCELoss(sigmoid=False)
        self.myLoss = nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=num_classes)
        # print(dir(self.efficientnet))
        # print(self.convnet)
        self.feature_maps = None
        # self.last_conv_layer = self.efficientnet.blocks[-1][-1].conv_pwl  # Replace with actual layer
        # self.last_conv_layer = self.efficientnet.blocks[-1][-1].conv_pwl
        self.last_conv_layer = self.efficientnet.conv_head
        self.last_conv_layer.register_forward_hook(self._hook_fn)
        # RuntimeError()
    
    def _hook_fn(self, module, input, output):
        self.feature_maps = output

    
    def forward(self, x):
        fwdPass = self.efficientnet(x)
        # fwdPass = self.convnet(x)
        return torch.sigmoid(fwdPass)


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.float()
        x_copy = copy.deepcopy(x)
        x = self(x)
        loss = self.myLoss(x, y)
        self.dice_weight = 0.00
        
        self.log('train_loss', loss)
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

        self.accuracy(x, y)
        combined_loss = (1 - self.dice_weight) * loss + self.dice_weight * avg_dice_loss
        # combined_loss = loss
        self.combinedLoss = combined_loss
        self.avg_dice = avg_dice_loss
        self.BCELoss = loss
        self.log("train_acc", self.accuracy)
        return combined_loss

    def training_epoch_end(self, outputs):
        # Clone the model and set it to eval mode after each epoch
        self.clone_model = EfficientNetClone()
        self.clone_model.load_state_dict(self.state_dict(), strict=False)
        self.clone_model.eval()
        if(self.current_epoch + 1) > 30:
            if (self.current_epoch + 1) % 10 == 0:
                self.dice_weight += 0.05
                print("Dice loss weight: ", self.dice_weight)
                print("Dice loss: ", self.avg_dice)
                print("BCE: ", self.BCELoss)
            

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = y.float()
        x = self(x)
        
        # loss = nn.BCEWithLogitsLoss()(x, y)
        loss = self.myLoss(x, y)
        
        self.log('val_loss', loss, sync_dist=True)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy, sync_dist=True)