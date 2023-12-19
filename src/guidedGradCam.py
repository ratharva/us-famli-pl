import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision import models, transforms
from PIL import Image
import numpy as np
from transferModel import EfficientNetTransfer

class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(ctx, input):
        positive_mask = (input > 0).type_as(input)
        ctx.save_for_backward(input, positive_mask)
        return input * positive_mask

    @staticmethod
    def backward(ctx, grad_output):
        input, positive_mask = ctx.saved_tensors
        grad_input = grad_output * positive_mask
        return grad_input

class GuidedGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def hook_fn(module, input, output):
            self.gradients = None

        target_layer = self.target_layer
        for _, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(GuidedBackpropReLU.backward)
            if str(module) == target_layer:
                module.register_forward_hook(hook_fn)

    def generate(self, input_image, target_class):
        input_image.requires_grad = True
        model_output = self.model(input_image)
        self.model.zero_grad()

        one_hot_output = torch.zeros_like(model_output)
        one_hot_output[0][target_class] = 1
        model_output.backward(gradient=one_hot_output)

        gradients = self.gradients[0]
        guided_gradients = gradients.cpu().numpy()[0]

        target_layer_output = self.target_layer_output.cpu().detach().numpy()[0]
        guided_gradcam = guided_gradients * target_layer_output
        guided_gradcam = np.maximum(guided_gradcam, 0)
        guided_gradcam = guided_gradcam / guided_gradcam.max()

        return guided_gradcam

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)

# Load pre-trained model
# model = models.resnet50(pretrained=True)
myNn = "efficientnet_b0"
myModel = "/mnt/raid/home/ayrisbud/train_output/classification/epoch=21-val_loss=1.06.ckpt"
model = EfficientNetTransfer(base_encoder=myNn, ckpt_path="/mnt/raid/C1_ML_Analysis/train_output/classification/extract_frames_blind_sweeps_c1_30082022_wscores_train_train_sample_clean_feat/epoch=9-val_loss=0.27.ckpt").load_from_checkpoint(myModel)
target_layer_name = model.efficientnet.convnet.features[8][0]  # Choose a target layer from the model architecture

# Load and preprocess the input image
image_path = "path_to_your_image.jpg"
input_image = preprocess_image(image_path)

# Initialize GuidedGradCAM
guided_gradcam = GuidedGradCAM(model, target_layer_name)

# Choose the target class index for which you want to generate the heatmap
target_class_index = 123  # Change this to your desired target class index

# Generate the Guided Grad-CAM heatmap
heatmap = guided_gradcam.generate(input_image, target_class_index)