import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributed import ReduceOp

import ultrasound_dataset as usd

import pytorch_lightning as pl

from monai.transforms import (
    Compose,
    LoadImageD,
    RandFlipD,
    RandRotateD,
    RandZoomD,
    ScaleIntensityD,    
    Lambda
)

from monai.networks.nets import AutoEncoder

# from azureml.core.run import Run
# run = Run.get_context()

class Feedforward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Feedforward, self).__init__()
        self.flat = nn.Flatten(start_dim=1)
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity
        self.tanh = nn.Tanh()
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):        
        shape = x.shape        
        x = self.flat(x)
        # Linear function
        out = self.fc1(x)
        # Non-linearity
        out = self.tanh(out)
        # Linear function (readout)
        out = self.fc2(out)
        out = out.reshape(shape)
        return out

class SaltAndPepper(nn.Module):    
    def __init__(self, prob=0.2):
        super(SaltAndPepper, self).__init__()
        self.prob = 0.2
    def forward(self, x):
        noise_tensor = torch.rand(x.shape).cuda()
        salt = torch.max(x)
        pepper = torch.min(x)
        x[noise_tensor < self.prob/2] = salt
        x[noise_tensor > 1-self.prob/2] = pepper
        return x


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
       
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def main(rank, args):

    ############################################################    
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(                                   
        backend='nccl',                                         
        init_method='env://',                                   
        world_size=world_size,
        rank=rank                                               
    )    

    device = torch.device(f"cuda:{rank}")

    torch.cuda.set_device(device)
    
    # model = AutoEncoder(
    #     spatial_dims=2,
    #     in_channels=3,
    #     out_channels=3,
    #     channels=(8, 16, 24, 32, 64, 128, 256),
    #     strides=(2, 2, 2, 2, 2, 2, 2),
    # )
    # model.intermediate = Feedforward(1024, 512)

    model = AutoEncoder(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        channels=(8, 16, 24, 32, 64, 128, 256),
        strides=(2, 2, 2, 2, 2, 2, 2)
    )
    
    model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # define loss function (criterion) and optimizer
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    intensity_transform = torch.nn.Sequential(
        transforms.RandomApply([transforms.ColorJitter(brightness=(0.3, 1.0), contrast=(0.3, 1.0), saturation=(0.3, 1.0), hue=(0, 0.4))], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
        SaltAndPepper(prob=0.15)
    )
    intensity_transform = intensity_transform.to(device)
    
    rotation_transform = torch.nn.Sequential(
        transforms.RandomApply([transforms.RandomResizedCrop(256, scale=(0.2, 1.))]),
        transforms.RandomApply([transforms.RandomHorizontalFlip()]),
        transforms.RandomApply([transforms.RandomRotation(degrees=(0,180))])
    )
    rotation_transform = rotation_transform.to(device)

    early_stop = EarlyStopping(patience=30, verbose=True, path=os.path.join(args.out, 'autoencoder.pt'))

    df_path = os.path.join(args.mount_point, 'CSV_files', 'extract_frames.csv')
    df = pd.read_csv(df_path)
    
    samples = int(len(df)*0.8)
    df_train = df.iloc[0:samples]

    train_dataset = usd.ITKImageDataset(df_train, mount_point=args.mount_point)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=24, prefetch_factor=4, pin_memory=True, sampler=train_sampler)

    df_val = df.iloc[samples:]
    val_dataset = usd.ITKImageDataset(df_val, mount_point=args.mount_point)    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=24, prefetch_factor=4, pin_memory=True, sampler=val_sampler)    

    max_steps = min(args.max_steps, len(train_loader))

    for epoch in range(0, args.epochs):

        model.train()
        
        for i, X in enumerate(train_loader):
            X = (X/255.0).to(device).unsqueeze(1).repeat(1,3,1,1)
            X = rotation_transform(X).contiguous()

            Y = model(intensity_transform(X))

            loss = loss_fn(X, Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0 and i%100 == 0:
                print("step", i+epoch*max_steps)
                print("loss", loss.item())
            #     run.log('step', i + epoch*max_steps)
            #     run.log('loss', loss.item())

            if i == max_steps:
                break

        model.eval()

        with torch.no_grad():
            
            running_loss = 0.0
            
            for i, X in enumerate(val_loader):
                X = (X/255.0).to(device).unsqueeze(1).repeat(1,3,1,1)
                X = rotation_transform(X).contiguous()

                Y = model(intensity_transform(X))

                loss = loss_fn(X, Y)

                running_loss += loss.item()                

            val_loss = torch.tensor([running_loss / len(val_loader)]).cuda()
            
            dist.all_reduce(val_loss, op=ReduceOp.SUM)
            val_loss = val_loss.cpu().item() / world_size
            if rank == 0:                
                # run.log('val_loss', val_loss)
                print('val_loss', val_loss)
                early_stop(val_loss, model.module)
                
                if early_stop.early_stop:
                    early_stop_indicator = torch.tensor([1.0]).cuda()
                else:
                    early_stop_indicator = torch.tensor([0.0]).cuda()
            else:
                early_stop_indicator = torch.tensor([0.0]).cuda()
            dist.all_reduce(early_stop_indicator, op=ReduceOp.SUM)
            if early_stop_indicator.cpu() == torch.tensor([1.0]):
                print("Early stopping")            
                break
        




if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Autoencoder Training')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount-point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--max-steps', help='Maximum number of steps', type=int, default=10000)
    parser.add_argument('--batch-size', help='Batch size', type=int, default=512)

    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'    
    os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())

    mp.spawn(main, nprocs=torch.cuda.device_count(), args=(args,))
