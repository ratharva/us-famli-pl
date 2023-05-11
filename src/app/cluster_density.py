
import os
import sys

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import pickle

import argparse


def main(args):

    test_df = pd.read_parquet(args.csv)
    # sns.set_theme(style="ticks")

    # Show the joint distribution using kernel density estimation
    g = sns.jointplot(
        data=test_df,
        x="pca_0", y="pca_1", hue=args.pred_column, palette=sns.color_palette("husl", test_df[args.pred_column].max() + 1),
        height=18,
        kind="kde",
        fill=True, 
        legend=False
    )    


    for cl in range(test_df[args.pred_column].max()):
        filtered_df = test_df.query('{pred_column} == {cl}'.format(pred_column=args.pred_column, cl=cl)).reset_index(drop=True)
        x = filtered_df['pca_0'].median()
        y = filtered_df['pca_1'].median()
        g.ax_joint.text(x, y, str(cl), fontsize='xx-large')


    out_name = os.path.splitext(args.csv)[0] + "_cluster_density.png"
    print("Writing:", out_name)
    plt.savefig(out_name) 
    


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Create image grids')    
    parser.add_argument('--csv', type=str, help='CSV file with cluster prediction', required=True)
    # parser.add_argument('--mount_point', type=str, help='Dataset mount point', default="./")
    parser.add_argument('--pred_column', type=str, help='Prediction/class column', default="pred_cluster")
    # parser.add_argument('--img_column', type=str, help='Image column', default="img_path")
    # parser.add_argument('--num_images', type=int, help='Image column', default=100)
    # parser.add_argument('--col_wrap', type=int, help='Image column', default=10)    
    # parser.add_argument('--out', type=str, help='Output directory', default="./out")    

    args = parser.parse_args()

    main(args)