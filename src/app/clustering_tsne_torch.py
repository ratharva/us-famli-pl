import os

import numpy as np
import pandas as pd

import torch
from kmeans_pytorch import kmeans

import pickle
import argparse

from sklearn.decomposition import PCA
# from tsne_torch import TorchTSNE as TSNE
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering


def add_pred(g):
    scores = np.array(g['score'])
    weights = scores/np.sum(scores)
    pred_g = np.sum(weights*np.array(g['pred']))
    g['pred_g'] = pred_g
    return g

def ga_pred_diff(g):
    g['pred_g_e'] = np.abs(g['abs_g'] - g['abs_e'])
    return g

def main(args):
    mount_point = args.mount_point
    csv_path = os.path.join(mount_point, args.csv)

    if(os.path.splitext(csv_path)[1] == ".csv"):        
        test_df = pd.read_csv(csv_path)
    else:
        test_df = pd.read_parquet(csv_path)

    test_df = test_df.groupby('study_id').apply(add_pred)

    test_df['abs_g'] = np.abs(test_df['pred_g'] - test_df['ga_boe'])
    test_df['abs_e'] = np.abs(test_df['ga_expert'] - test_df['ga_boe'])

    test_df = test_df.groupby('study_id').apply(ga_pred_diff)

    features_path = os.path.join(mount_point, args.features)

    with open(features_path, 'rb') as f:
        features = pickle.load(f)

    if args.query is not None:
        test_df = test_df.query(args.query)
        features = features[test_df.index]
        test_df = test_df.reset_index(drop=True)

    split = args.split

    # pca = PCA(n_components=args.n_components)
    # pca.fit(features)

    # print(pca.explained_variance_ratio_)
    # print(pca.singular_values_)

    if args.split_fn:
        idx_sample = list(pd.read_csv(args.split_fn, header=None)[1])
    else:
        idx_sample = np.random.choice(np.arange(features.shape[0]), size=int(features.shape[0]*split))    

    test_df = test_df.loc[idx_sample].reset_index(drop=True)
    features = features[idx_sample]

    # features_transformed = torch.from_numpy(pca.transform(features))

    
    tsne = TSNE(n_components=2, perplexity=args.perplexity, n_iter=1000, verbose=True, random_state=123, learning_rate='auto')    
    features_embedded = tsne.fit_transform(features)

    out_tsne = features_path.replace(".pickle", "_tsne.pickle")
    pickle.dump(tsne, open(out_tsne,"wb"))

    cluster_ids_x, cluster_centers = kmeans(
        X=torch.from_numpy(features_embedded), num_clusters=args.n_clusters, distance='euclidean', device=torch.device('cuda:0')
    )

    # clustering = SpectralClustering(
    #         n_clusters=args.n_clusters,
    #         eigen_solver="arpack",
    #         affinity="nearest_neighbors",
    #         n_neighbors=20,
    #         random_state=0, 
    #         verbose=True
    #     )
    # clustering.fit(features_embedded)

    test_df["pred_cluster"] = cluster_ids_x
    test_df["pca_0"] = features_embedded[:,0]
    test_df["pca_1"] = features_embedded[:,1]

    out_csv = features_path.replace(".pickle", "_sample.csv")
    print("Writing:", out_csv)
    test_df.to_csv(out_csv, index=False)

    out_features = features_path.replace(".pickle", "_sample.pickle")
    print("Writing:", out_features)
    with open(out_features, 'wb') as f:
        pickle.dump(features, f)

    out_centers = features_path.replace(".pickle", "_sample_centers.pickle")
    print("Writing:", out_centers)
    with open(out_centers, 'wb') as f:
        pickle.dump(cluster_centers.cpu().numpy(), f)


    
    out_centers = features_path.replace(".pickle", "_sample_centers.pickle")
    print("Writing:", out_centers)
    with open(out_centers, 'wb') as f:
        pickle.dump(cluster_centers.cpu().numpy(), f)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Autoencoder Training')    
    parser.add_argument('--csv', type=str, help='CSV file for clustering', required=True)    
    parser.add_argument('--features', type=str, help='Features path', required=True)
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--elbow', help='Perform elbow analysis for kmeans and identify the optimal number of clusters', type=bool, default=False)
    parser.add_argument('--n_clusters', help='Number of clusters', type=int, default=10)
    # parser.add_argument('--n_components', help='n_components parameter for PCA, check sklearn decomposition for more info', type=float, default=0.99)
    parser.add_argument('--split_fn', help='Text file single column with indices to grab from features and dataframe', type=str, default=None)
    parser.add_argument('--split', help='Split the sample', type=float, default=0.1)
    parser.add_argument('--query', help='Query to filter the dataframe', type=str, default=None)
    parser.add_argument('--perplexity', help='Perplexity for TSNE', type=int, default=300)


    args = parser.parse_args()

    main(args)