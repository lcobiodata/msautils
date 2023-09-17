#!/usr/bin/env python
"""
Protlearn is a module that implements the DB[RC/S3] (Density Based Residue Clustering by Dissimilarity
Between Sequence SubSets) methodology.

Copyright (C) 2023, Lucas Carrijo de Oliveira (lucas@ebi.ac.uk)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import re
from prince import MCA
import matplotlib.cm as cm
from Bio.SeqUtils import seq3
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import OPTICS
import networkx as nx
from itertools import combinations
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import to_tree
from operator import itemgetter
import fastcluster
from scipy.cluster.hierarchy import fcluster
from functools import lru_cache
import random
import weblogo
from weblogo import LogoOptions, LogoData, LogoFormat  # , write
import os

# Download necessary resources from NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Set a seed value for random number generation
seed_value = 42  # You can choose any integer value as the seed

# Use the seed value to initialize the random number generator
random.seed(seed_value)

# Define a dictionary for amino acid stereochemistry
STHEREOCHEMISTRY = {
    'Aliphatic': ['G', 'A', 'V', 'L', 'I'],
    'Amide': ['N', 'Q'],
    'Aromatic': ['F', 'Y', 'W'],
    'Basic': ['H', 'K', 'R'],
    'Big': ['M', 'I', 'L', 'K', 'R'],
    'Hydrophilic': ['R', 'K', 'N', 'Q', 'P', 'D'],
    'Median': ['E', 'V', 'Q', 'H'],
    'Negatively charged': ['D', 'E'],
    'Non-polar': ['F', 'G', 'V', 'L', 'A', 'I', 'P', 'M', 'W'],
    'Polar': ['Y', 'S', 'N', 'T', 'Q', 'C'],
    'Positively charged': ['K', 'R'],
    'Similar (Asn or Asp)': ['N', 'D'],
    'Similar (Gln or Glu)': ['Q', 'E'],
    'Small': ['C', 'D', 'P', 'N', 'T'],
    'Tiny': ['G', 'A', 'S'],
    'Very hydrophobic': ['L', 'I', 'F', 'W', 'V', 'M'],
    'With hydroxyl': ['S', 'T', 'Y'],
    'With sulfur': ['C', 'M']
}


def generate_wordcloud(df, column='Protein names'):
    """
    Generate a word cloud visualization from protein names in a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame containing protein names.
        column (str, default='Protein names'): The name of the column in the DataFrame containing protein names.

    This function extracts substrate and enzyme names from the specified column using regular expressions, normalizes the names, and creates a word cloud plot.
    """
    # Extract the substrate and enzyme names using regular expressions
    matches = df[column].str.extract(r'(.+?) ([\w\-,]+ase)', flags=re.IGNORECASE)
    # String normalization pipeline
    df['Substrate'] = matches[0] \
        .fillna('') \
        .apply(lambda x: '/'.join(re.findall(r'\b(\w+(?:ene|ine|ate|yl))\b', x, flags=re.IGNORECASE))) \
        .apply(lambda x: x.lower())
    df['Enzyme'] = matches[1] \
        .fillna('') \
        .apply(lambda x: x.split('-')[-1] if '-' in x else x) \
        .apply(lambda x: x.lower())
    df['Label'] = df['Substrate'].str.cat(df['Enzyme'], sep=' ').str.strip()
    df = df.copy()
    # Plot the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(
        WordCloud(
            width=800,
            height=400,
            background_color='white'
        ).generate(
            ' '.join(
                sorted(
                    set([string for string in df.Label.values.tolist() if len(string) > 0])
                )
            )
        ),
        interpolation='bilinear'
    )
    plt.axis('off')
    plt.show()
    return None


def get_newick(node, parent_dist, leaf_names, newick=""):
    """
    Recursively construct a Newick string representation of a hierarchical tree.

    Parameters:
        node (TreeNode): The current node in the hierarchical tree.
        parent_dist (float): The distance between the current node and its parent node.
        leaf_names (list): A list of leaf names corresponding to the tree nodes.
        newick (str, default=""): The Newick string representation of the hierarchical tree.

    Returns:
        A Newick string representation of the hierarchical tree.
    """
    if node.is_leaf():
        return f"{leaf_names[node.id]}:{parent_dist - node.dist:.2f}{newick}"
    else:
        if len(newick) > 0:
            newick = f"):{parent_dist - node.dist:.2f}{newick}"
        else:
            newick = ");"
        newick = get_newick(node.get_left(), node.dist, leaf_names, newick)
        newick = get_newick(node.get_right(), node.dist, leaf_names, f",{newick}")
        newick = f"({newick}"
        return newick


def write_dendrogram(linkage_object, headers, prefix='output'):
    """
    Write the Newick representation of a dendrogram to a file.

    Parameters:
        linkage_object (object): The linkage object resulting from hierarchical clustering.
        headers (list): A list of headers corresponding to the data points being clustered.
        prefix (str, default='output'): The prefix for the output dendrogram file.

    This function writes the Newick representation of a dendrogram to a file using the provided linkage object and headers.
    """
    tree = to_tree(linkage_object, False)
    with open(f'{prefix}_dendrogram.nwk', 'w') as outfile:
        outfile.write(get_newick(tree, "", tree.dist, headers))


def cached_property(method):
    """
    Create a cached property using the lru_cache decorator.

    Parameters:
        method (callable): A method that calculates the property value.

    Returns:
        The cached property value.
    """

    @property
    @lru_cache()
    def wrapper(self):
        return method(self)

    return wrapper


class MSA(pd.DataFrame):
    """
    A class for processing and analyzing Multiple Sequence Alignments (MSA).
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the MSA object.
        """
        super().__init__(*args, **kwargs)
        self.msa_file = None
        self.headers = None
        self.raw_data = None
        self.data = None
        self.alphabet = None
        self.analysis = None
        self.coordinates = None
        self.labels = None
        self.sorted_importance = None
        self.selected_features = None
        self.clusters = None

    def parse_msa_file(self, msa_file, msa_format="fasta", *args, **kwargs):
        """
        Parse an MSA file and store the raw data in the MSA object.

        Parameters:
            msa_file (str): The path to the MSA file.
            msa_format (str, optional): The format of the MSA file (default is "fasta").
        """
        self.msa_file = msa_file
        headers, sequences = [], []
        for record in SeqIO.parse(self.msa_file, msa_format, *args, **kwargs):
            headers.append(record.id)
            sequences.append(record.seq)
        self.headers = headers
        self.raw_data = pd.DataFrame(np.array(sequences), index=headers)

    def cleanse_data(self, indel='-', remove_lowercase=True, threshold=.9, plot=False):
        """
        Cleanse the MSA data by removing columns and rows with missing values.

        Parameters:
            indel (str, optional): The character representing gaps/indels (default is '-').
            remove_lowercase (bool, optional): Whether to remove lowercase characters (default is True).
            threshold (float, optional): The threshold for missing values (default is 0.9).
            plot (bool, optional): Whether to plot a heatmap of missing values (default is False).
        """
        to_remove = [indel]
        if remove_lowercase:
            clean = self.raw_data.copy()
            to_remove += [chr(i) for i in range(ord('a'), ord('z') + 1)]
        else:
            clean = self.raw_data.applymap(str.upper).copy()
        clean.replace(to_remove, np.nan, inplace=True)
        min_rows = int(threshold * clean.shape[0])
        # Remove columns with NaN values above the threshold
        clean.dropna(thresh=min_rows, axis=1, inplace=True)
        min_cols = int(threshold * clean.shape[1])
        # Remove rows with NaN values above the threshold
        clean.dropna(thresh=min_cols, axis=0, inplace=True)
        if plot:
            # Plot the heatmap
            sns.heatmap(clean.isna().astype(int), cmap='binary', xticklabels=False, yticklabels=False, cbar=False)
            # Show the plot
            plt.show()
        self.data = clean.reset_index(drop=True) \
            .drop_duplicates() \
            .fillna('-') \
            .copy()

    def analyse(self, plot=False, *args, **kwargs):
        """
        Perform Multidimensional Correspondence Analysis (MCA) on the MSA data.

        Parameters:
            plot (bool, optional): Whether to plot the results (default is False).
        """
        # Perform MCA
        mca = MCA(*args, **kwargs)
        mca.fit(self.data)
        if plot:
            try:
                # Plot together the scatter plot of both sequences and residues overlaid
                mca.plot(
                    self.data,
                    x_component=0,
                    y_component=1
                )
            finally:
                pass
        self.analysis = mca

    def reduce(self):
        """
        Reduce the dimensionality of the MSA data using the MCA analysis.
        """
        # Get the row coordinates
        self.coordinates = self.analysis.transform(self.data)

    def get_labels(self, plot=False):
        """
        Cluster the MSA data and obtain cluster labels.

        Parameters:
            plot (bool, optional): Whether to plot the clustering results (default is False).
        """
        coordinates = np.array(self.coordinates)
        # Define a range of potential number of clusters to evaluate
        min_clusters = 3
        max_clusters = 10
        # Perform clustering for different numbers of clusters and compute silhouette scores
        silhouette_scores = []
        for k in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, n_init=10)  # Set n_init explicitly
            kmeans.fit(coordinates)
            labels = kmeans.labels_
            score = silhouette_score(coordinates, labels)
            silhouette_scores.append(score)
        # Find the best number of clusters based on the highest silhouette score
        best_num_clusters = np.argmax(silhouette_scores) + min_clusters
        # Perform clustering with the best number of clusters
        kmeans = KMeans(n_clusters=best_num_clusters, n_init=10)  # Set n_init explicitly
        kmeans.fit(coordinates)
        if plot:
            cluster_centers = kmeans.cluster_centers_
            # Plot the scatter plot colored by clusters
            plt.scatter(coordinates[:, 0], coordinates[:, 1], c=kmeans.labels_, cmap='viridis')
            plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', label='Cluster Centers')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.title('Scatter Plot - Clusters')
            plt.legend()
            plt.show()
        self.labels = kmeans.labels_

    @staticmethod
    def henikoff(data):
        """
        Calculate sequence weights using the Henikoff & Henikoff algorithm.

        Parameters:
            data (pd.DataFrame): The MSA data.

        Returns:
            pd.Series: A Series containing the calculated weights.
        """
        data_array = data.to_numpy()  # Convert DataFrame to NumPy array
        size, length = data_array.shape
        weights = []
        for seq_index in range(size):
            row = data_array[seq_index, :]
            unique_vals, counts = np.unique(row, return_counts=True)
            k = len(unique_vals)
            matrix_row = 1. / (k * counts)
            weights.append(np.sum(matrix_row) / length)
        return pd.Series(weights, index=data.index)

    def select_features(self, n_estimators=None, random_state=None, plot=False):
        """
        Select important features (residues) from the MSA data.

        Parameters:
            n_estimators (int, optional): Parameter n_estimators for RandomForest.
            random_state: (int, optional): Parameter random_state for RandomForest.
            plot (bool, optional): Whether to plot feature selection results (default is False).
        """
        x = self.data
        y = pd.get_dummies(self.labels).astype(int) if len(
            set(self.labels)
        ) > 2 else self.labels
        # Perform one-hot encoding on the categorical features
        encoder = OneHotEncoder()
        x_encoded = encoder.fit_transform(x)
        # Get the column names for the encoded features
        encoded_feature_names = []
        for i, column in enumerate(x.columns):
            categories = encoder.categories_[i]
            for category in categories:
                feature_name = f'{column}_{category}'
                encoded_feature_names.append(feature_name)
        # Convert X_encoded to DataFrame
        x_encoded_df = pd.DataFrame.sparse.from_spmatrix(x_encoded, columns=encoded_feature_names)
        # Create and train the Random Forest classifier
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        rf.fit(x_encoded_df, y)
        # Feature selection
        feature_selector = SelectFromModel(rf, threshold='median')
        feature_selector.fit_transform(x_encoded_df, y)
        selected_feature_indices = feature_selector.get_support(indices=True)
        selected_features = x_encoded_df.columns[selected_feature_indices]
        # Calculate feature importances for original columns
        sorted_importance = pd.DataFrame(
            {
                'Residues': selected_features,
                'Importance': rf.feature_importances_[selected_feature_indices],
                'Columns': map(lambda ftr: int(ftr.split('_')[0]), selected_features)
            }
        )[['Columns', 'Importance']].groupby('Columns').sum()['Importance'].sort_values(ascending=False)
        sorted_features = sorted_importance.index
        if plot:
            fig, ax1 = plt.subplots(figsize=(16, 4))
            # Bar chart of percentage importance
            xvalues = range(len(sorted_features))
            ax1.bar(xvalues, sorted_importance, color='b')
            ax1.set_ylabel('Summed Importance', fontsize=12)
            ax1.tick_params(axis='y', labelsize=12)
            # Line chart of cumulative percentage importance
            ax2 = ax1.twinx()
            ax2.plot(xvalues, np.cumsum(sorted_importance) / np.sum(sorted_importance), color='r', marker='.')
            ax2.set_ylabel('Cumulative Importance', fontsize=12)
            ax2.tick_params(axis='y', labelsize=12)
            # Rotate x-axis labels
            plt.xticks(xvalues, sorted_features, fontsize=12)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
            plt.show()
        self.selected_features = selected_features
        self.sorted_importance = sorted_importance

    def get_clusters(self, selected_columns=None, threshold=0.9, plot=False, export_tree=False, prefix='output'):
        """
        Cluster sequences in the MSA data.

        Parameters:
            selected_columns (list, optional): list of previously selected MSA columns to base on.
            threshold (float, optional): The threshold for selecting important residues (default is 0.9).
            plot (bool, optional): Whether to plot the clustering results (default is False).
            export_tree (bool, optional): Whether to export the clustering dendrogram (default is False).
            prefix (str, optional): The prefix for the output dendrogram file (default is 'output').
        """
        if not selected_columns:
            # Perform feature selection on data
            self.select_features()
            # Calculate cumulative sum of importance
            cumulative_importance = np.cumsum(self.sorted_importance) / np.sum(self.sorted_importance)
            # Find the index where cumulative importance exceeds or equals 0.9
            index = np.where(cumulative_importance >= threshold)[0][0]
            # Get the values from sorted_features up to the index
            selected_columns = self.sorted_importance.index[:index + 1].values
            # Filter the selected features to get the most important residues
        selected_residues = [x for x in self.selected_features if int(x.split('_')[0]) in selected_columns]
        df_res = self.analysis.column_coordinates(self.data[selected_columns]).loc[selected_residues]
        # Get sequence weights through Henikoff & Henikoff algorithm
        weights = self.henikoff(self.data[selected_columns])
        # Create an empty graph
        g = nx.Graph()
        for idx in df_res.index:
            col, aa = idx.split('_')
            col = int(col)
            rows = self.data[selected_columns].index[self.data[selected_columns][col] == aa].tolist()
            # Filter and sum values based on valid indices
            p = weights.iloc[[i for i in rows if i < len(weights)]].sum()
            # Add a node with attributes
            g.add_node(
                f'{seq3(aa)}{col}',
                idx=idx,
                aa=aa,
                col=col,
                coord=(
                    df_res.loc[idx, 0],
                    df_res.loc[idx, 1]
                ),
                rows=rows,
                p=p
            )
        nodelist = sorted(g.nodes(), key=lambda x: g.nodes[x]['p'], reverse=True)
        df_res = df_res.loc[[g.nodes[u]['idx'] for u in nodelist]]
        df_res.columns = ['x_mca', 'y_mca']
        df_res = df_res.copy()
        # Generate pairwise combinations
        pairwise_comparisons = list(combinations(g.nodes, 2))
        # Add edges to the graph based on pairwise calculation of Jaccard's dissimilarity (1 - similarity)
        for u, v in pairwise_comparisons:
            asymmetric_distance = set(g.nodes[u]['rows']) ^ set(g.nodes[v]['rows'])
            union = set(g.nodes[u]['rows']) | set(g.nodes[v]['rows'])
            weight = float(
                weights.iloc[[i for i in list(asymmetric_distance) if i < len(weights)]].sum()
            ) / float(
                weights.iloc[[i for i in list(union) if i < len(weights)]].sum()
            ) if g.nodes[u]['col'] != g.nodes[v]['col'] else 1.
            g.add_edge(
                u,
                v,
                weight=weight
            )
        # Generate a distance matrix
        d = nx.to_numpy_array(g, nodelist=nodelist)
        # Apply OPTICS on the points
        optics = OPTICS(metric='precomputed', min_samples=3)
        optics.fit(d)
        # Retrieve cluster labels
        cluster_labels = optics.labels_
        unique_labels = np.unique(cluster_labels)
        clusters = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(unique_labels):
            cluster_indices = np.where(cluster_labels == cluster_label)[0]
            cluster_nodes = [nodelist[idx] for idx in cluster_indices]
            clusters[i] = cluster_nodes
        adhesion = []
        for row_idx in self.data.index:
            temp = []
            for cluster in clusters:
                count = 0
                for node in cluster:
                    if self.data.loc[row_idx, g.nodes[node]['col']] == g.nodes[node]['aa']:
                        count += 1
                temp.append(float(count) / float(len(cluster)))
            adhesion.append(np.array(temp))
        adhesion = np.array(adhesion)
        df_adh = pd.DataFrame(adhesion)
        df_adh.columns = [f"Cluster {label + 1}" if label >= 0 else "Noise" for label in unique_labels]
        itemgetter_func = itemgetter(*self.data.index)
        seq_ids = itemgetter_func(self.raw_data.index)
        df_adh.index = seq_ids
        # Run clustermap with potential performance improvement
        z = fastcluster.linkage(df_adh, method='ward')
        if export_tree:
            write_dendrogram(z, prefix)
        # List to store silhouette scores
        silhouette_scores = []
        # Define a range of possible values for k
        k_values = range(2, 10)
        # Calculate silhouette score for each value of k
        for k in k_values:
            labels = fcluster(z, k, criterion='maxclust')
            silhouette_scores.append(silhouette_score(df_adh, labels))
        # Find the index of the maximum silhouette score
        best_index = np.argmax(silhouette_scores)
        # Get the best value of k
        best_k = k_values[best_index]
        # Get the cluster labels for each sequence in the MSA
        sequence_labels = fcluster(z, best_k, criterion='maxclust')
        if plot:
            # fig = plt.figure(figsize=(12, 7))  # Create the main figure container
            # Plot the distance matrix
            fig, ax = plt.subplots()
            im = ax.imshow(d, cmap='viridis')
            # Add a colorbar
            ax.figure.colorbar(im, ax=ax)
            plt.show()
            ordering = optics.ordering_
            # Perform MDS to obtain the actual points
            mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress='auto')
            points = mds.fit_transform(d)
            df_res['x_mds'] = points[:, 0]
            df_res['y_mds'] = points[:, 1]
            df_res['label'] = cluster_labels
            df_res = df_res.copy()
            # Plot the ordering analysis
            plt.figure(figsize=(12, 4))
            plt.bar(range(len(ordering)), ordering, width=1., color='black')
            plt.xlim([0, len(ordering)])
            plt.xlabel('Points')
            plt.ylabel('Reachability Distance')
            plt.title('Ordering Analysis')
            # Show the plots
            plt.show()
            # Create a figure with two subplots
            fig, axs = plt.subplots(1, 2, figsize=(12, 7))
            # Plot for MCA
            axs[0].set_xlabel('MCA Dimension 1', fontsize=14)
            axs[0].set_ylabel('MCA Dimension 2', fontsize=14)
            axs[0].set_title('Residues from MCA Colored by Cluster', fontsize=16)
            # Plot for MDS
            axs[1].set_xlabel('MDS Dimension 1', fontsize=14)
            axs[1].set_ylabel('MDS Dimension 2', fontsize=14)
            axs[1].set_title('Residues from MDS Colored by Cluster', fontsize=16)
            # Plot on the subplots
            x_mca = df_res['x_mca']
            y_mca = df_res['y_mca']
            x_mds = df_res['x_mds']
            y_mds = df_res['y_mds']
            labels = df_res['label']
            unique_labels = np.unique(labels)
            color_map = cm.get_cmap('tab10', len(unique_labels))
            handles = []
            all_labels = []
            for i, cluster_label in enumerate(unique_labels):
                if cluster_label == -1:
                    color = 'grey'
                    marker = 'x'
                    label = 'Noise'
                    alpha = 0.25
                else:
                    color = color_map(i)
                    marker = 'o'
                    label = f'Cluster {cluster_label + 1}'
                    alpha = None
                scatter = axs[0].scatter(
                    x_mca[labels == cluster_label],
                    y_mca[labels == cluster_label],
                    color=color,
                    marker=marker,
                    label=label,
                    alpha=alpha
                )
                axs[1].scatter(
                    x_mds[labels == cluster_label],
                    y_mds[labels == cluster_label],
                    color=color,
                    marker=marker,
                    label=label,
                    alpha=alpha
                )
                # Collect the scatter plot handles and labels
                handles.append(scatter)
                all_labels.append(label)
            # Rearrange labels to fill one row first and then the second row
            n_cols = 6  # Number of columns in the legend
            all_labels_reordered = [all_labels[i::n_cols] for i in range(n_cols)]
            all_labels_reordered = sum(all_labels_reordered, [])  # Flatten the nested list
            # Create a single legend for both subplots with reordered labels
            fig.legend(handles, all_labels_reordered, bbox_to_anchor=(0.5, 0.0), loc='upper center', borderaxespad=0,
                       ncol=n_cols, fontsize=12)
            # Adjust the layout to accommodate the legend
            fig.tight_layout(rect=[0, 0.05, 1, 0.9])
            cbar_kws = {"orientation": "horizontal"}
            g = sns.clustermap(df_adh.drop('Noise', axis=1), col_cluster=False, yticklabels=False,
                               cbar_pos=(.4, .9, .4, .02), cbar_kws=cbar_kws, figsize=(5, 5))
            # Show the plot
            plt.show()
        self.clusters = {label: cluster for label, cluster in zip(unique_labels, clusters)}
        self.labels = sequence_labels

    # def bootstrap(self):
    #     """
    #     :return:
    #     """
    #     return None

    # def merged_to(self, metadata_file, *args, **kargs):
    #     """
    #     EXAMPLE of how to merge data frames (alignment + metadata)
    #     Join the dataframes based on the 'ID' column
    #
    #     :return:
    #     """
    #     return pd.merge(
    #         pd.merge(
    #             pd.read_csv(metadata_file, *args, **kargs),
    #             pd.DataFrame(
    #                 {
    #                     'Entry':[self.index[idx].split('/')[0].split('_')[0] for idx in self.index],
    #                     'Index':list(self.index)
    #                 }
    #             ),
    #             on='Entry'
    #         ),
    #         self,
    #         left_on='Index',
    #         right_index=True,
    #         how='inner'
    #     ).copy()

    # def get_alphabet(self, data=None):
    #     if data is None:
    #         data = self.data
    #     self.alphabet = np.unique(data.applymap(str.upper).values.flatten())
