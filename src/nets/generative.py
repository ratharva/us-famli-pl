import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision 

import pytorch_lightning as pl

import torchmetrics

from generative.inferers import LatentDiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks import nets
from generative.networks.schedulers import DDPMScheduler
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from monai import transforms
from monai.networks import normal_init
from monai.networks.nets import Generator, Discriminator

class GaussianNoise(nn.Module):    
    def __init__(self, mean=0.0, std=0.05):
        super(GaussianNoise, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def forward(self, x):
        if self.training:
            return x + torch.normal(mean=self.mean, std=self.std, size=x.size(), device=x.device)
        return x
    
class RandCoarseShuffle(nn.Module):    
    def __init__(self, prob=0.75, holes=16, spatial_size=32):
        super(RandCoarseShuffle, self).__init__()
        self.t = transforms.RandCoarseShuffle(prob=prob, holes=holes, spatial_size=spatial_size)
    def forward(self, x):
        if self.training:
            return self.t(x)
        return x

class SaltAndPepper(nn.Module):    
    def __init__(self, prob=0.05):
        super(SaltAndPepper, self).__init__()
        self.prob = prob
    def __call__(self, x):
        noise_tensor = torch.rand(x.shape)
        salt = torch.max(x)
        pepper = torch.min(x)
        x[noise_tensor < self.prob/2] = salt
        x[noise_tensor > 1-self.prob/2] = pepper
        return x

class VQVAEPL(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.autoencoderkl = nets.VQVAE(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(64, 128, 256, 512, 1024),
            num_res_channels=512,
            num_res_layers=2,
            downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
            upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
            num_embeddings=256,
            embedding_dim=128
        )

        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

        self.discriminator = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        

        self.adversarial_loss = PatchAdversarialLoss(criterion="least_squares")        

        # For mixed precision training
        # self.scaler_g = GradScaler()
        # self.scaler_d = GradScaler()

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise(0.0, 0.05),
            RandCoarseShuffle(),
            SaltAndPepper()
            
        )
        

    def configure_optimizers(self):
        optimizer_g = optim.AdamW(self.autoencoderkl.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        optimizer_d = optim.AdamW(self.discriminator.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]

    def training_step(self, train_batch, batch_idx):
        x = train_batch

        optimizer_g, optimizer_d = self.optimizers()
        
        optimizer_g.zero_grad()

        # with autocast(enabled=True):
        #     reconstruction, z_mu, z_sigma = self.autoencoderkl(x)

        #     recons_loss = F.l1_loss(reconstruction.float(), x.float())
        #     p_loss = self.perceptual_loss(reconstruction.float(), x.float())
        #     kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        #     kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        #     loss_g = recons_loss + (self.hparams.kl_weight * kl_loss) + (self.hparams.perceptual_weight * p_loss)

        #     if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
        #         logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
        #         generator_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
        #         loss_g += self.hparams.adversarial_weight * generator_loss

        # self.scaler_g.scale(loss_g).backward()
        # self.scaler_g.step(optimizer_g)
        # self.scaler_g.update()

        reconstruction, z_mu, z_sigma = self.autoencoderkl(self.noise_transform(x))

        recons_loss = F.l1_loss(reconstruction.float(), x.float())
        p_loss = self.perceptual_loss(reconstruction.float(), x.float())
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss_g = recons_loss + (self.hparams.kl_weight * kl_loss) + (self.hparams.perceptual_weight * p_loss)

        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()
        
        loss_d = 0
        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:

            # with autocast(enabled=True):
            #     optimizer_d.zero_grad()

            #     logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            #     loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
            #     logits_real = self.discriminator(x.contiguous().detach())[-1]
            #     loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            #     discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            #     loss_d = self.hparams.adversarial_weight * discriminator_loss

            # self.scaler_d.scale(loss_d).backward()
            # self.scaler_d.step(optimizer_d)
            # self.scaler_d.update()
            
            optimizer_d.zero_grad()

            logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = self.discriminator(x.contiguous().detach())[-1]
            loss_d_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            loss_d = self.hparams.adversarial_weight * discriminator_loss

            loss_d.backward()
            optimizer_d.step()

        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

        return {"train_loss_g": loss_g, "train_loss_d": loss_d}
        

    def validation_step(self, val_batch, batch_idx):
        x = val_batch

        # with autocast(enabled=True):
        #     reconstruction, z_mu, z_sigma = self.autoencoderkl(x)
        #     recon_loss = F.l1_loss(x.float(), reconstruction.float())

        reconstruction, z_mu, z_sigma = self.autoencoderkl(x)
        recon_loss = F.l1_loss(x.float(), reconstruction.float())

        self.log("val_loss", recon_loss, sync_dist=True)

        


    def forward(self, images):        
        return self.autoencoderkl(images)

class VQVAEPL(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.autoencoderkl = nets.VQVAE(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(64, 128, 256, 512, 1024),
            num_res_channels=512,
            num_res_layers=2,
            downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
            upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
            num_embeddings=256,
            embedding_dim=128
        )

        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

        self.discriminator = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        

        self.adversarial_loss = PatchAdversarialLoss(criterion="least_squares")        

        # For mixed precision training
        # self.scaler_g = GradScaler()
        # self.scaler_d = GradScaler()

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise(0.0, 0.05),
            RandCoarseShuffle(),
            SaltAndPepper()
            
        )
        

    def configure_optimizers(self):
        optimizer_g = optim.AdamW(self.autoencoderkl.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        optimizer_d = optim.AdamW(self.discriminator.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]

    def training_step(self, train_batch, batch_idx):
        x = train_batch

        optimizer_g, optimizer_d = self.optimizers()
        
        optimizer_g.zero_grad()

        # with autocast(enabled=True):
        #     reconstruction, z_mu, z_sigma = self.autoencoderkl(x)

        #     recons_loss = F.l1_loss(reconstruction.float(), x.float())
        #     p_loss = self.perceptual_loss(reconstruction.float(), x.float())
        #     kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        #     kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        #     loss_g = recons_loss + (self.hparams.kl_weight * kl_loss) + (self.hparams.perceptual_weight * p_loss)

        #     if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
        #         logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
        #         generator_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
        #         loss_g += self.hparams.adversarial_weight * generator_loss

        # self.scaler_g.scale(loss_g).backward()
        # self.scaler_g.step(optimizer_g)
        # self.scaler_g.update()

        reconstruction, z_mu, z_sigma = self.autoencoderkl(self.noise_transform(x))

        recons_loss = F.l1_loss(reconstruction.float(), x.float())
        p_loss = self.perceptual_loss(reconstruction.float(), x.float())
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss_g = recons_loss + (self.hparams.kl_weight * kl_loss) + (self.hparams.perceptual_weight * p_loss)

        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()
        
        loss_d = 0
        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:

            # with autocast(enabled=True):
            #     optimizer_d.zero_grad()

            #     logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            #     loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
            #     logits_real = self.discriminator(x.contiguous().detach())[-1]
            #     loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            #     discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            #     loss_d = self.hparams.adversarial_weight * discriminator_loss

            # self.scaler_d.scale(loss_d).backward()
            # self.scaler_d.step(optimizer_d)
            # self.scaler_d.update()
            
            optimizer_d.zero_grad()

            logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = self.discriminator(x.contiguous().detach())[-1]
            loss_d_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            loss_d = self.hparams.adversarial_weight * discriminator_loss

            loss_d.backward()
            optimizer_d.step()

        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

        return {"train_loss_g": loss_g, "train_loss_d": loss_d}
        

    def validation_step(self, val_batch, batch_idx):
        x = val_batch

        # with autocast(enabled=True):
        #     reconstruction, z_mu, z_sigma = self.autoencoderkl(x)
        #     recon_loss = F.l1_loss(x.float(), reconstruction.float())

        reconstruction, z_mu, z_sigma = self.autoencoderkl(x)
        recon_loss = F.l1_loss(x.float(), reconstruction.float())

        self.log("val_loss", recon_loss, sync_dist=True)

        


    def forward(self, images):        
        return self.autoencoderkl(images)


class GanDiff(pl.LightningModule):
    def __init__(self, diffusion, autoencoder, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        
        self.post_quant_conv = autoencoder.post_quant_conv
        self.decoder = autoencoder.decoder

        self.diffusion = diffusion

        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.inferer = DiffusionInferer(scheduler=scheduler)

        # self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

        self.discriminator = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=3, out_channels=3)        

        self.adversarial_loss = PatchAdversarialLoss(criterion="least_squares")



    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
        

    def configure_optimizers(self):
        optimizer_g = optim.AdamW(self.decoder.parameters() + self.post_quant_conv.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        optimizer_d = optim.AdamW(self.discriminator.parameters(),
                                lr=self.hparams.lr_d,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]

    def training_step(self, train_batch, batch_idx):

        X = train_batch

        optimizer_g, optimizer_d = self.optimizers()
        
        optimizer_g.zero_grad()


        fake = self(X.shape[0])
        
        logits_fake = self.discriminator(fake.contiguous().float())[-1]
        generator_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
        loss_g = generator_loss 

        loss_g.backward()
        optimizer_g.step()       

        loss_d = 0.
        
        optimizer_d.zero_grad()

        logits_fake = self.discriminator(fake.contiguous().detach())[-1]
        loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.discriminator(X.contiguous().detach())[-1]
        loss_d_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = discriminator_loss

        loss_d.backward()
        optimizer_d.step()

        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

        return {"train_loss_g": loss_g, "train_loss_d": loss_d}

    def validation_step(self, val_batch, batch_idx):

        X = val_batch

        fake = self(X.shape[0])
        
        logits_fake = self.discriminator(fake.contiguous().float())[-1]
        generator_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
        loss_g = generator_loss 

        logits_fake = self.discriminator(fake.contiguous().detach())[-1]
        loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.discriminator(X.contiguous().detach())[-1]
        loss_d_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = discriminator_loss

        self.log("val_loss_g", loss_g, sync_dist=True)
        self.log("val_loss_d", loss_d, sync_dist=True)
        self.log("val_loss", loss_d + loss_g, sync_dist=True)


    def forward(self, num):

        noise = torch.randn((num, 1, 64, 64))
        noise = noise.to(self.device)

        z = self.inferer.sample(
            input_noise=noise, diffusion_model=self.diffusion, scheduler=self.scheduler, save_intermediates=False
        )
        
        return self.decode(z)

class Gan(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False

        self.generator = Generator(
            latent_shape=self.hparams.emb_dim,
            start_shape=(64, 8, 8),
            channels=[32, 16, 8, 1],
            strides=[2, 2, 2, 1],
        )

        # self.discriminator = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)
        self.discriminator = Discriminator(
            in_shape=(1, 64, 64),
            channels=(8, 16, 32, 64, 1),
            strides=(2, 2, 2, 2, 1),
            num_res_units=1,
            kernel_size=5,
        )

        self.adversarial_loss = PatchAdversarialLoss(criterion="least_squares")

        self.resize_transform = transforms.Resize([-1, 64, 64])
        

    def configure_optimizers(self):
        optimizer_g = optim.AdamW(self.generator.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        optimizer_d = optim.AdamW(self.discriminator.parameters(),
                                lr=self.hparams.lr_d,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]

    def training_step(self, train_batch, batch_idx):

        X = self.resize_transform(train_batch)

        optimizer_g, optimizer_d = self.optimizers()
        
        optimizer_g.zero_grad()


        fake = self(X.shape[0])
        
        logits_fake = self.discriminator(fake.contiguous().float())[-1]
        generator_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
        loss_g = generator_loss 

        loss_g.backward()
        optimizer_g.step()
        
        
        optimizer_d.zero_grad()

        logits_fake = self.discriminator(fake.contiguous().detach())[-1]
        loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.discriminator(X.contiguous().detach())[-1]
        loss_d_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = discriminator_loss

        loss_d.backward()
        optimizer_d.step()

        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

        return {"train_loss_g": loss_g, "train_loss_d": loss_d}

    def validation_step(self, val_batch, batch_idx):

        X = self.resize_transform(val_batch)

        fake = self(X.shape[0])
        
        logits_fake = self.discriminator(fake.contiguous().float())[-1]
        generator_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
        loss_g = generator_loss 

        logits_fake = self.discriminator(fake.contiguous().detach())[-1]
        loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.discriminator(X.contiguous().detach())[-1]
        loss_d_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = discriminator_loss

        self.log("val_loss_g", loss_g, sync_dist=True)
        self.log("val_loss_d", loss_d, sync_dist=True)
        self.log("val_loss", loss_d + loss_g, sync_dist=True)


    def forward(self, num):

        latent = torch.randn(num, self.hparams.emb_dim).to(self.device)
        
        return self.generator(latent)