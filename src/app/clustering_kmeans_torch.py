import os

import numpy as np
import pandas as pd

import torch
from kmeans_pytorch import kmeans

import pickle
import argparse

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def main(args):
    mount_point = args.mount_point
    csv_path = os.path.join(mount_point, args.csv)

    if(os.path.splitext(csv_path)[1] == ".csv"):        
        test_df = pd.read_csv(csv_path)
    else:
        test_df = pd.read_parquet(csv_path)

    features_path = os.path.join(mount_point, args.features)

    with open(features_path, 'rb') as f:
        features = pickle.load(f)

    if args.split_fn:
        idx_sample = list(pd.read_csv(args.split_fn, header=None)[1])
    else:
        idx_sample = np.random.choice(np.arange(features.shape[0]), size=int(features.shape[0]*args.split))

    print("Sample:", len(idx_sample))

    test_df = test_df.loc[idx_sample].reset_index(drop=True)
    features = features[idx_sample]

    if args.query is not None:
        test_df = test_df.query(args.query)
        features = features[test_df.index]
        test_df = test_df.reset_index(drop=True)

    cluster_centers = None
    if args.cluster_centers:
        cluster_centers = torch.from_numpy(np.array(pd.read_csv(args.cluster_centers, header=None)))

    # kmeans
    cluster_ids_x, cluster_centers = kmeans(
        X=torch.from_numpy(features), num_clusters=args.n_clusters, distance='euclidean', device=torch.device('cuda:0'), cluster_centers=cluster_centers
    )

    test_df["pred_cluster"] = cluster_ids_x

    out_csv = features_path.replace(".pickle", "kmeans_sample.csv")
    print("Writing:", out_csv)
    test_df.to_csv(out_csv, index=False)

    out_features = features_path.replace(".pickle", "kmeans_sample.pickle")
    print("Writing:", out_features)
    with open(out_features, 'wb') as f:
        pickle.dump(features, f)

    out_centers = features_path.replace(".pickle", "kmeans_sample_centers.pickle")
    print("Writing:", out_centers)
    with open(out_centers, 'wb') as f:
        pickle.dump(cluster_centers.cpu().numpy(), f)
    

    if args.tsne:

        split_tsne = ""

        if args.split_tnse:
            idx_sample = np.random.choice(np.arange(features.shape[0]), size=int(features.shape[0]*args.split_tsne))
            test_df = test_df.loc[idx_sample].reset_index(drop=True)
            features = features[idx_sample]

            split_tsne = "tsne_" + str(args.split_tsne)

        features_embedded = TSNE(n_components=2, perplexity=300, random_state=123).fit_transform(features)    
        test_df["pca_0"] = features_embedded[:,0]
        test_df["pca_1"] = features_embedded[:,1]

        out_csv = features_path.replace(".pickle", split_tsne + "_sample.csv")
        print("Writing:", out_csv)
        test_df.to_csv(out_csv, index=False)

        out_features = features_path.replace(".pickle", split_tsne + "_sample.pickle")
        print("Writing:", out_features)
        with open(out_features, 'wb') as f:
            pickle.dump(features, f)

        out_centers = features_path.replace(".pickle", split_tsne + "_sample_centers.pickle")
        print("Writing:", out_centers)
        with open(out_centers, 'wb') as f:
            pickle.dump(cluster_centers.cpu().numpy(), f)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Clustering with torch implementation of kmeans')    
    parser.add_argument('--csv', type=str, help='CSV file for clustering', required=True)    
    parser.add_argument('--features', type=str, help='Features path', required=True)
    parser.add_argument('--tsne', type=int, help='Apply TSNE', default=1)
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--elbow', help='Perform elbow analysis for kmeans and identify the optimal number of clusters', type=bool, default=False)
    parser.add_argument('--n_clusters', help='Number of clusters', type=int, default=10)
    parser.add_argument('--cluster_centers', help='CSV_files with cluster centers', type=str, default=None)
    parser.add_argument('--split', help='Split the sample', type=float, default=1.0)
    parser.add_argument('--split_tsne', help='Split the sample for TSNE', type=float, default=None)
    parser.add_argument('--split_fn', help='Text file single column with indices to grab from features and dataframe', type=str, default=None)
    parser.add_argument('--query', help='Query to filter the dataframe', type=str, default=None)


    args = parser.parse_args()

    main(args)