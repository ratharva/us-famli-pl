import pytorch_lightning as pl
import torch

# Lightning Module for YOLOv5
class YOLOv5LightningModel(pl.LightningModule):

    def __init__(self, model_name='yolov5s', pretrained=True):
        super(YOLOv5LightningModel, self).__init__()
        
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=pretrained)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        results = self(images)
        loss = results.losses["total_loss"]
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        results = self(images)
        val_loss = results.losses["total_loss"]
        self.log('val_loss', val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer