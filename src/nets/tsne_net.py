import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision 

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import TensorDataset, DataLoader

class TSNE_klmse(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.hparams.emb_dim, self.hparams.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hparams.hidden_dim, 2),
        )

        self.mse = nn.MSELoss()
        self.kl = F.kl_div


    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(),
                                lr=self.hparams.lr,
                                momentum=self.hparams.momentum,
                                weight_decay=self.hparams.weight_decay)
        return optimizer
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
        #                                                     T_max=self.hparams.args.epochs,
        #                                                     eta_min=self.hparams.args.lr/50)
        # return [optimizer], [lr_scheduler]        


    def compute_loss(self, X_low_dim_model, X_low_dim_tsne):

        # Y = X_low_dim_tsne
        loss = 0

        if "kl" in self.hparams.loss:
            probs_tsne = self.calculate_probs(X_low_dim_tsne)
            probs_model = self.calculate_probs(X_low_dim_model)

            loss = loss + self.hparams.kl_w*self.kl(torch.log(probs_model), probs_tsne, reduction='sum')

        if "mse" in self.hparams.loss:
            loss = loss + self.hparams.mse_w*self.mse(X_low_dim_tsne, X_low_dim_model)

        return loss

    # Function to calculate pairwise distances and convert them to probabilities.
    def calculate_probs(self, X):
        # Calculate pairwise distances.
        X_squared = (X ** 2).sum(dim=-1)
        dot_product = X @ X.t()
        distances = X_squared.unsqueeze(1) - 2.0 * dot_product + X_squared.unsqueeze(0)
        # Ensure diagonal elements are zero (distance from a point to itself).
        distances = distances - torch.diag(distances.diag())
        # Convert distances to probabilities using Gaussian kernel.
        sigma = distances.mean().sqrt()
        probabilities = torch.exp(-distances / (2 * sigma ** 2))
        # Normalize the distances so they sum 1
        probabilities = probabilities/probabilities.sum()
        return probabilities

    def training_step(self, batch, batch_idx):
        X, Y = batch
        x = self(X)
        loss = self.compute_loss(x, Y)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        x = self(X)
        loss = self.compute_loss(x, Y)

        self.log('val_loss', loss, sync_dist=True)

    def forward(self, X):
        return self.model(X)

