
import os
import sys

# module_path = os.path.abspath(os.path.join(__file__, '..', '..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

# from loaders.ultrasound_dataset import USDataset
# from transforms.ultrasound_transforms import Moco2TrainTransforms, Moco2EvalTransforms, AutoEncoderTrainTransforms

# from torchvision import transforms

# import plotly.express as px
# import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import SimpleITK as sitk
import numpy as np

import pickle

import argparse
from mpl_toolkits.axes_grid1 import ImageGrid
import math

# from monai.transforms import (    
#     AsChannelFirst,
#     AddChannel,
#     Compose,    
#     RandFlip,
#     RandRotate,
#     CenterSpatialCrop,
#     ScaleIntensityRange,
#     RandAdjustContrast,
#     RandGaussianNoise,
#     RandGaussianSmooth
# )


def main(args):
    # if not os.path.exists(args.out):
    #     os.makedirs(args.out)

    test_df = pd.read_parquet(args.csv)

    clusters_range = [test_df[args.pred_column].min(), test_df[args.pred_column].max()]

    num_images = args.col*args.row

    for cl in range(clusters_range[0], clusters_range[1] + 1):


        filtered_df = test_df.query('{pred_column} == {cl}'.format(pred_column=args.pred_column, cl=cl)).reset_index(drop=True)

        filtered_df = filtered_df.sample(n=min(num_images, len(filtered_df))).reset_index(drop=True)

        imgs = []    

        # if not os.path.exists(os.path.join(args.out, str(cl))):
        #     os.makedirs(os.path.join(args.out, str(cl)))

        for idx, row in filtered_df.iterrows():
            img_path = row[args.img_column]    
            img_path = os.path.join(args.mount_point, img_path)

            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))

            imgs.append(img)

            # out_name = os.path.join(args.out, str(cl), str(idx) + ".png")

            # fig = plt.imshow(img, cmap='gray')
            # plt.savefig(out_name)

        fig = plt.figure(figsize=args.fig_size)

        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(args.row, args.col)
                 )


        for ax, im in zip(grid, imgs):        
            ax.imshow(im, cmap='gray')

        out_name = os.path.splitext(args.csv)[0] + "_cluster{cl}_image_grid.png".format(cl=cl)
        print("Writing:", out_name)
        fig.savefig(out_name) 
    


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Create image grids')    
    parser.add_argument('--csv', type=str, help='CSV file with cluster prediction', required=True)
    parser.add_argument('--mount_point', type=str, help='Dataset mount point', default="./")
    parser.add_argument('--pred_column', type=str, help='Prediction/class column', default="pred_cluster")
    parser.add_argument('--img_column', type=str, help='Image column', default="img_path")
    parser.add_argument('--row', type=int, help='Row length (number of images per row)', default=10)
    parser.add_argument('--col', type=int, help='Col length (number of images per column)', default=10)    
    parser.add_argument('--fig_size', nargs="+", type=int, help='Figure size', default=[24, 24])        

    args = parser.parse_args()

    main(args)