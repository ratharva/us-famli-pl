import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

import pickle
import argparse

def create_linkage_matrix(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    return linkage_matrix


# iris = load_iris()
# X = iris.data

def main(args):

    mount_point = args.mount_point

    csv_path = os.path.join(mount_point, args.csv)

    test_df = pd.read_csv(csv_path)

    features_path = os.path.join(mount_point, args.features)

    with open(features_path, 'rb') as f:
        features = pickle.load(f)

    pca = PCA(n_components=0.95)
    pca.fit(features)

    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

    features_transformed = pca.transform(features)


    split = args.split
    idx_sample = np.random.choice(np.arange(features.shape[0]), size=int(features.shape[0]*split))

    test_df = test_df.loc[idx_sample].reset_index(drop=True)
    features_transformed = features_transformed[idx_sample]

    print(features_transformed.shape)

    # setting distance_threshold=0 ensures we compute the full tree.
    # model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    # model = model.fit(features_transformed)
    # linkage_matrix = create_linkage_matrix(model, truncate_mode="level", p=4)


    out_csv = features_path.replace(".pickle", "_sample_hierarchical.csv")
    print("Writing:", out_csv)
    test_df.to_csv(out_csv, index=False)

    out_features = features_path.replace(".pickle", "_sample_hierarchical.pickle")
    print("Writing:", out_features)
    with open(out_features, 'wb') as f:
        pickle.dump(features_transformed, f)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Autoencoder Training')    
    parser.add_argument('--csv', type=str, help='CSV file for clustering', required=True)    
    parser.add_argument('--features', type=str, help='Features path', required=True)
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    # parser.add_argument('--clustering', help='Type of clustering', type=str, default="SpectralClustering")
    parser.add_argument('--n_clusters', help='Number of clusters', type=int, default=10)
    parser.add_argument('--n_components', help='n_components parameter for PCA, check sklearn decomposition for more info', type=float, default=0.95)
    parser.add_argument('--split', help='Split the sample', type=float, default=0.1)


    args = parser.parse_args()

    main(args)