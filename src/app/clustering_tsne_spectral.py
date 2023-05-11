import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import pickle
import argparse

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
            

    # pca = PCA(n_components=0.95)
    # pca.fit(features)

    # print(pca.explained_variance_ratio_)
    # print(pca.singular_values_)

    if args.split_fn:
        idx_sample = list(pd.read_csv(args.split_fn, header=None)[0])
    else:
        idx_sample = np.random.choice(np.arange(features.shape[0]), size=int(features.shape[0]*args.split))

    test_df = test_df.loc[idx_sample].reset_index(drop=True)
    features = features[idx_sample]

    # features_transformed = pca.transform(features)
    features_transformed = features

    print(features_transformed.shape)

    if args.clustering == "TSNE":        
        features_embedded = TSNE(n_components=2, perplexity=300, random_state=123).fit_transform(features_transformed)
        clustering = SpectralClustering(
            n_clusters=args.n_clusters,
            eigen_solver="arpack",
            affinity="nearest_neighbors",
            n_neighbors=20,
            random_state=0, 
            verbose=True
        )
        clustering.fit(features_embedded)

    test_df["pred"] = clustering.labels_
    test_df["pca_0"] = features_embedded[:,0]
    test_df["pca_1"] = features_embedded[:,1]


    out_csv = features_path.replace(".pickle", "_sample.csv")
    print("Writing:", out_csv)
    test_df.to_csv(out_csv, index=False)

    out_features = features_path.replace(".pickle", "_sample.pickle")
    print("Writing:", out_features)
    with open(out_features, 'wb') as f:
        pickle.dump(features, f)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Spectral clustering on the TSNE embeddings')    
    parser.add_argument('--csv', type=str, help='CSV file for clustering', required=True)    
    parser.add_argument('--features', type=str, help='Features path', required=True)
    parser.add_argument('--n_clusters', help='Number of clusters', type=int, default=8)
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    parser.add_argument('--clustering', help='Type of clustering', type=str, default="TSNE")    
    parser.add_argument('--split', help='Split the sample', type=float, default=0.1)
    parser.add_argument('--split_fn', help='Text file single column with indices to grab from features and dataframe', type=str, default=None)


    args = parser.parse_args()

    main(args)