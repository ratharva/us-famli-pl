
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
from seaborn._statistics import KDE
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

import pandas as pd
import SimpleITK as sitk
import numpy as np

import pickle

import argparse
from scipy import stats

from sklearn.neighbors import KernelDensity
from scipy.special import softmax 


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
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    test_df = pd.read_parquet(args.csv)
    # sns.set_theme(style="ticks")

    filtered_df = test_df.query('{pred_column} == {cl}'.format(pred_column=args.pred_column, cl=args.n_cluster)).reset_index(drop=True)

    print(filtered_df.shape)

    # density, support = KDE()(x1=filtered_df["pca_0"], x2=filtered_df["pca_1"])

    model = KernelDensity(kernel='gaussian', bandwidth='scott')

    x_train = np.stack([filtered_df["pca_0"], filtered_df["pca_1"]], axis=1)

    model.fit(x_train)
    log_dens = model.score_samples(x_train)

    filtered_df["density"] = np.exp(log_dens)

    sns.scatterplot(filtered_df, x='pca_0', y='pca_1', hue='density', linewidth=0, alpha = 0.7)

    print(filtered_df["density"].describe())


    filtered_df_q75 = filtered_df.query('density > {q}'.format(q=filtered_df["density"].quantile(q=0.75)))
    # print(filtered_df_q75)

    imgs = []    

    out_dir = os.path.join(args.out, str(args.n_cluster) + '_q{q}'.format(q=0.75))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for idx, row in filtered_df_q75.iterrows():
        img_path = row[args.img_column]    
        img_path = os.path.join(args.mount_point, img_path)

        img = sitk.ReadImage(img_path)
        # img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))

        out_name = os.path.join(out_dir, str(idx) + ".png")        

        sitk.WriteImage(img, out_name)
        # fig = plt.imshow(img, cmap='gray')
        # plt.savefig(out_name)



    # print(np.array(support).shape)
    
    # neigh = NearestNeighbors(n_neighbors=1)
    # neigh.fit(np.array(support).transpose())


    # idx = neigh.kneighbors(np.stack([filtered_df["pca_0"], filtered_df["pca_1"]], axis=1), return_distance=False)

    # print(np.array(idx).shape)


    

    # Z = np.reshape(kernel(positions).T, X.shape)

    


    out_name = os.path.join(args.out, "cluster_density_{cl}.png".format(cl=args.n_cluster))    
    plt.savefig(out_name) 
    


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Create image grids')    
    parser.add_argument('--csv', type=str, help='CSV file for clustering', required=True)
    parser.add_argument('--n_cluster', type=int, help='Cluster number', required=True)
    parser.add_argument('--mount_point', type=str, help='Dataset mount point', default="./")
    parser.add_argument('--pred_column', type=str, help='Prediction/class column', default="pred_cluster")
    parser.add_argument('--img_column', type=str, help='Image column', default="img_path")    
    
    parser.add_argument('--col_wrap', type=int, help='Image column', default=10)    
    parser.add_argument('--out', type=str, help='Output directory', default="./out")



    args = parser.parse_args()

    main(args)