#!/usr/bin/env python
"""
Protlearn is a module that applies Machine Learning techniques to extract meaningful information from
Multiple Sequence Alignments (MSA) of homologous protein families.

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
from scipy.ndimage import zoom
from wordcloud import WordCloud
import nltk
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import re
from prince import MCA
from kneed import KneeLocator
import networkx as nx
from Bio.SeqUtils import seq3
from sklearn.metrics import silhouette_score
import fastcluster
from scipy.cluster.hierarchy import fcluster, dendrogram
from collections import defaultdict
import logomaker as lm
import os
import sys
import argparse
import pickle


def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


# Obtain the viridis colormap
my_palette = plt.cm.get_cmap('viridis')


class MSA(pd.DataFrame):
    """
    A class for processing and analyzing Multiple Sequence Alignments (MSA).
    """

    def __init__(self, msa_file=None, metadata_file=None, msa_format="fasta", metadata=None, *args, **kwargs):
        """
        Initialize the MSA object.

        Args:
            *args: Variable-length positional arguments.
            **kwargs: Variable-length keyword arguments.
        """
        headers = []
        sequences = []
        if msa_file:
            try:
                # Parse the MSA file and extract headers and sequences
                for record in SeqIO.parse(msa_file, msa_format, *args, **kwargs):
                    headers.append(record.id)
                    sequences.append(record.seq)

            except FileNotFoundError:
                raise ValueError(f"The file '{msa_file}' was not found.")

            except ValueError as ve:
                # This exception might be raised if the provided format is incorrect
                raise ValueError(f"Error parsing the file '{msa_file}' with format '{msa_format}'. Details: {ve}")

            except Exception as e:
                # General error handler for unexpected exceptions
                raise RuntimeError(f"An unexpected error occurred while parsing the file '{msa_file}'. Details: {e}")

            data, index = np.array(sequences), headers
            # Initialize the DataFrame part of the MSA class
            super().__init__(data=data, index=index, *args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
        self.metadata_file = metadata_file
        self.metadata = None
        self.positions_map = None  # Mapping of positions in the MSA
        self.dirty = None  # Dirty data (before cleansing)
        self.clean = None  # Clean data (after cleansing)
        self.unique = None  # Processed MSA data
        self.mca = None  # MCA object containing Multidimensional Correspondence Analysis results
        self.coordinates = None  # Reduced data coordinates
        self.tree = None
        self.labels = None  # Sequence labels or clusters
        self.encoded = None  # One-hot encoded data set
        self.sequence_order = None
        self.residue_order = None
        self.sorted_importance = None  # Sorted feature importance
        self.selected_features = None  # Selected features
        self.selected_columns = None  # Selected columns from the data
        self.profiles = None  # Residue profiles
        self.wordcloud_data = None  # Word cloud data
        self.logos_data = None  # Logo data
        # Find correspondence between MSA colums and residue positions in each sequence
        self.map_positions()

    def map_positions(self):
        """
        Map residue positions in the Multiple Sequence Alignment (MSA) based on sequence headers.

        This method calculates the position of each residue in the MSA sequences and stores
        the mapping in the 'positions_map' attribute of the MSA object. The mapping is
        based on the sequence headers, which typically include information about the
        sequence's starting position.

        The 'positions_map' is a dictionary with sequence headers as keys and a sub-dictionary
        as values. The sub-dictionary contains MSA columns (keys) and their corresponding
        positions (values).

        For example:
        {
            'Seq1/1-100': {21: 1, 25: 2, ...},
            'Seq2/171-432': {101: 171, 103: 172, ...},
            ...
        }

        This mapping is useful for downstream analysis that requires knowing the position
        of residues in the MSA.

        Note:
        - Residues represented by '-' (indels/gaps) are not included in the mapping.

        Example:
        msa = MSA('example.fasta')
        msa.map_positions()

        Access the mapping:
        positions = msa.positions_map
        """
        if self.empty:
            warnings.warn("The MSA data frame is empty. Ensure you've loaded the data correctly.", UserWarning)
            return

        # Initialize a dictionary to store residue positions
        self.positions_map = defaultdict(dict)

        # Iterate through sequence headers and sequences
        for header in self.index:
            if re.search(r'.+/\d+-\d+', header):
                sequence = self.loc[header]

                # Extract the starting position from the header
                offset, _ = header.split('/')[1].split('-')
                position = int(offset) - 1  # offset

                # Iterate through residues in the sequence
                for index, value in zip(sequence.index, sequence.values):
                    if value != '-':
                        position += 1
                        # Store the residue position in the positions_map dictionary
                        self.positions_map[header][index] = position

    def _plot_cleanse_heatmaps(self, save=False, show=False):
        """
        Generate and display cleansing heatmaps plot on the specified axes.
        """
        # Create a figure with two subplots (heatmaps)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Create the heatmap before cleansing on the first Axes
        ax1 = axes[0]  # First subplot
        heatmap_before = ax1.imshow(self.dirty.isna().astype(int), cmap='viridis', aspect='auto', extent=[0, 1, 0, 1])

        # Adjust the position of the first subplot
        ax1.set_position([0.05, 0.1, 0.4, 0.8])  # [left, bottom, width, height]

        # Create the heatmap after cleansing on the second Axes
        ax2 = axes[1]  # Second subplot
        heatmap_after = ax2.imshow(self.clean.isna().astype(int), cmap='viridis', aspect='auto', extent=[0, 1, 0, 1])

        # Adjust the position of the second subplot
        ax2.set_position([0.55, 0.1, 0.4, 0.8])  # [left, bottom, width, height]

        # Create a color bar axis between the two heatmaps
        cax = fig.add_axes([0.46, 0.1, 0.02, 0.8])  # [left, bottom, width, height]

        # Add a color bar with the color bar axis
        cbar = plt.colorbar(heatmap_before, cax=cax)
        cbar.set_label('Gaps (Indels)')

        ax1.axis('off')  # Turn off axis labels for the first subplot
        ax2.axis('off')  # Turn off axis labels for the second subplot

        if save:
            plt.savefig("./output/cleanse_heatmaps.png", dpi=300)

        if show:
            plt.show()
        else:
            plt.close()

    def cleanse(self, indel='-', remove_lowercase=True, threshold=.9, plot=False, save=False, show=False):
        """
        Cleanse the MSA data by removing columns and rows with gaps/indels.

        Parameters:
            indel (str, optional): The character representing gaps/indels (default is '-').
            remove_lowercase (bool, optional): Whether to remove lowercase characters (default is True).
            threshold (float, optional): The threshold for gaps/indels (default is 0.9).
            plot (bool, optional): Whether to plot a heatmap of gaps/indels (default is False).
            save (bool, optional): Whether to save a heatmap of gaps/indels (default is False).
            show (bool, optional): Whether to show a heatmap of gaps/indels (default is False).
        """
        if self.empty:
            raise ValueError("The MSA data frame is empty. Ensure you've loaded the data correctly.")

        # Define characters to remove based on parameters
        to_remove = [indel]

        # Create a copy of the raw data as the 'dirty' data
        self.dirty = self.copy()

        # Check if lowercase characters should be removed
        if remove_lowercase:
            to_remove += [chr(i) for i in range(ord('a'), ord('z') + 1)]
        else:
            self.dirty = self.applymap(str.upper).copy()

        # Replace specified characters with NaN
        self.dirty.replace(to_remove, np.nan, inplace=True)
        # print(f"self.dirty.shape after replacing specified characters with NaN: {self.dirty.shape}")

        # Create a copy of 'dirty' data as the 'clean' data
        self.clean = self.dirty.copy()
        # print(f"self.clean.shape right after creating a copy of 'dirty' data as the 'clean' data: {self.clean.shape}")

        # Calculate the minimum number of non-NaN values for rows
        min_rows = int(threshold * self.clean.shape[0])
        # print(f"min_rows: {min_rows}")

        # Remove columns with NaN values above the threshold
        self.clean.dropna(thresh=min_rows, axis=1, inplace=True)
        # print(f"self.clean.shape after removing columns with NaN values above the threshold: {self.clean.shape}")

        # Calculate the minimum number of non-NaN values for columns
        min_cols = int(threshold * self.clean.shape[1])
        # print(f"min_cols: {min_cols}")

        # Remove rows with NaN values above the threshold
        self.clean.dropna(thresh=min_cols, axis=0, inplace=True)
        # print(f"self.clean.shape after removing rows with NaN values above the threshold: {self.clean.shape}")

        # Reset the index, drop duplicates, and fill NaN values with '-'
        self.unique = self.clean.reset_index(drop=True) \
            .drop_duplicates() \
            .fillna('-') \
            .copy()
        # print(f"self.unique.shape after removing duplicate rows: {self.unique.shape}")

        # If plotting is enabled, plot heatmaps
        if plot:
            self._plot_cleanse_heatmaps(save=save, show=show)

    def _plot_scree(self, elbow=None, save=False, show=True):
        """
        Plots the scree plot of the squared eigenvalues of self.mca.eigenvalues_.

        Parameters:
            save (bool, optional): If True, the scree plot will be saved to a file.
            show (bool, optional): If True, the scree plot will be displayed.
        """
        squared_eigenvalues = np.square(self.mca.eigenvalues_)
        max_dim = min(len(squared_eigenvalues), 20)

        plt.figure(figsize=(12, 4))
        plt.plot(range(1, max_dim + 1), squared_eigenvalues[:max_dim], marker='o', linestyle='-', color=my_palette(0.2))
        if elbow is not None:
            plt.axvline(x=elbow, color=my_palette(0.8), linestyle='--')
        plt.xlabel('Dimensions', fontsize=14)
        plt.xticks(ticks=range(1, max_dim + 1))
        plt.ylabel('Variance Explained (Eigenvalue\u00b2)', fontsize=14)
        # plt.title('Scree Plot')

        if save:
            plt.savefig("./output/scree_plot_squared_eigenvalues.png")

        if show:
            plt.show()
        else:
            plt.close()

    def _plot_perceptual_map(self, processed=False, save=False, show=True):
        """
        Generate and display a perceptual map of residues, either selected or all.

        Parameters:
            processed (bool, optional): Determines the type of plot. If True, visualizes selected residues with clustering.
                                        If False, visualizes all sequences and residues without any selection or labeling.
                                        Default is True.
            save (bool, optional): If True, the generated perceptual map will be saved to a file. Default is False.
            show (bool, optional): If True, the generated perceptual map will be displayed immediately. Default is True.

        Notes:
        - This method is designed to visualize the residues on a perceptual map.
        - The perceptual map can offer insights into the distribution, clustering, or relationships of the residues based on certain metrics or dimensions.
        - This is particularly useful for understanding the spatial arrangement or similarity of residues in some context.
        """
        # Create figure
        plt.figure(figsize=(8, 6))

        if processed:
            selected_residues = [
                feature for feature in self.selected_features if int(feature.split('_')[0]) in self.selected_columns
            ]
            df_res = self.mca.column_coordinates(self.unique[self.selected_columns]).loc[selected_residues]
            residues = []
            for res_idx in df_res.index:
                msa_column, amino_acid = res_idx.split('_')
                three_letter_aa = seq3(amino_acid)
                residues.append(three_letter_aa + msa_column)
            df_res = df_res.set_index(pd.Index(residues))

            # Create legends for cluster labels
            unique_labels = np.unique(self.labels)
            legend_handles = []
            # Iterate through unique labels
            for label in unique_labels:
                indices = np.where(self.labels == label)[0]
                plt.scatter(
                    self.coordinates[indices, 0],
                    self.coordinates[indices, 1],
                    color=my_palette(label / len(unique_labels)),
                    alpha=0.5
                )
                legend_handles.append(
                    plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}', markersize=10,
                               markerfacecolor=my_palette(label / len(unique_labels)))
                )

            # Scatter plot of labeled residues
            plt.scatter(df_res[0], df_res[1], marker='*', color='black', alpha=0.5, s=50)

            # Annotate labeled residues
            for i, (x, y) in enumerate(zip(df_res[0], df_res[1])):
                plt.annotate(df_res.index[i], (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

            legend_handles.append(
                # plt.Line2D([0], [0], marker='*', color='k', alpha=0.5, label='Selected Residues', markersize=10)
                plt.scatter([0], [0], marker='*', color='black', alpha=0.5, label='Selected Residues')
            )

            plt.legend(handles=legend_handles, title='Sequence Clusters')

        else:  # if processed is False
            # Scatter plot for sequences
            plt.scatter(self.coordinates[:, 0], self.coordinates[:, 1], marker='o', color=my_palette(0.5), alpha=0.5,
                        label='Sequences')

            # Scatter plot for all residues
            df_res_all = self.mca.column_coordinates(self.unique)
            plt.scatter(df_res_all[0], df_res_all[1], marker='*', color='black', alpha=0.5, s=50, label='Residues')
            plt.legend()

        # Set labels, title, and grid
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid()

        if save:
            plt.savefig(f"./output/perceptual_map_{'after' if processed else 'before'}.png")

        if show:
            plt.show()
        else:
            plt.close()

    def reduce(self, n_components=None, criterion='explained_variance', explained_variance_threshold=0.95, plot=False,
               save=False,
               show=False, *args, **kwargs):
        """
        Perform Multidimensional Correspondence Analysis (MCA) on the MSA data to reduce dimensionality.

        Parameters: n_components (int, optional): Number of dimensions to retain. If None, it will be determined
        based on the criterion. criterion (str, optional): Criterion to decide on the number of dimensions. Options:
        'scree', 'explained_variance'. Default is 'scree'. explained_variance_threshold (float, optional): Threshold
        for cumulative explained variance, used when criterion='explained_variance'. Default is 0.95. plot (bool,
        optional): If True, the MCA results will be plotted (default is False). save (bool, optional): If True and
        `plot` is also True, the plotted results will be saved to a file. show (bool, optional): If True and `plot`
        is also True, the plotted results will be displayed. *args, **kwargs: Additional arguments and keyword
        arguments passed to the MCA.

        Notes:
        - Multidimensional Correspondence Analysis (MCA) is used to reduce the dimensionality of the MSA data.
        - The MCA results are stored in the 'mca' attribute of the MSA object.
        - The row coordinates after reduction are stored in the 'coordinates' attribute.

        Example:
        msa = MSA('example.fasta')
        msa.map_positions()
        msa.cleanse()
        msa.reduce(plot=True, save=True) # This will plot the results and save the plot.
        """
        # Check if unique exists and is not None
        if not hasattr(self, "unique") or self.unique is None:
            self.cleanse()

        # Perform MCA
        self.mca = MCA(n_components=min(self.unique.shape[0], self.unique.shape[1]) - 1,
                       *args, **kwargs)
        self.mca.fit(self.unique)

        # Decide on the number of dimensions if n_components is None
        if n_components is None:
            squared_eigenvalues = np.square(self.mca.eigenvalues_)

            if criterion == 'scree':
                # Use the KneeLocator to find the elbow
                kn = KneeLocator(range(1, len(squared_eigenvalues) + 1), squared_eigenvalues,
                                 curve='convex', direction='decreasing')
                n_components = kn.elbow

            elif criterion == 'explained_variance':
                cumulative_var = np.cumsum(squared_eigenvalues) / np.sum(squared_eigenvalues)
                n_components = np.argmax(cumulative_var > explained_variance_threshold) + 1

        transformed_data = self.mca.transform(self.unique)
        self.coordinates = transformed_data.iloc[:, :n_components].values

        if plot:
            self._plot_scree(elbow=n_components, save=save, show=show)
            self._plot_perceptual_map(processed=False, save=save, show=show)

    def _plot_wordclouds(self, column='Protein names', save=False, show=False):
        """
        Generate and plot word cloud visualizations from protein names in a DataFrame.

        Parameters:
            metadata (str, default=None): Path to metadata file in tsv format.
            column (str, default='Protein names'): The name of the column in the DataFrame containing protein names.
            save (bool, default=False): Whether to save word clouds.
            show (bool, default=False): Whether to show word clouds.

        This method extracts substrate and enzyme names from the specified column using regular expressions,
        normalizes the names, and creates word cloud plots for each cluster of sequences.
        """

        if self.metadata_file is not None:

            # Read the TSV file into a DataFrame
            try:
                self.metadata = pd.read_csv(self.metadata_file, delimiter='\t')
            except FileNotFoundError:
                raise

            self.wordcloud_data = {}

            # Process data for each cluster
            for label in np.unique(self.labels):
                indices = self.unique.iloc[self.labels == label].index
                headers = self.index[indices]
                entry_names = [header.split('/')[0] for header in headers]

                result = self.metadata[self.metadata['Entry Name'].isin(entry_names)].copy()

                matches = result[column].str.extract(r'(.+?) ([\w\-,]+ase)', flags=re.IGNORECASE)

                result['Substrate'] = matches[0] \
                    .fillna('') \
                    .apply(lambda x: '/'.join(re.findall(r'\b(\w+(?:ene|ine|ate|yl))\b', x, flags=re.IGNORECASE))) \
                    .apply(lambda x: x.lower())
                result['Enzyme'] = matches[1] \
                    .fillna('') \
                    .apply(lambda x: x.split('-')[-1] if '-' in x else x) \
                    .apply(lambda x: x.lower())
                result['Label'] = result['Substrate'].str.cat(result['Enzyme'], sep=' ').str.strip()
                result = result.copy()

                wordcloud_text = ' '.join(
                    sorted(
                        set([string for string in result.Label.values.tolist() if len(string) > 0])
                    )
                )
                self.wordcloud_data[label] = wordcloud_text

            # Plotting logic
            num_clusters = len(self.wordcloud_data)
            fig, axs = plt.subplots(num_clusters, 1, figsize=(8, 4 * num_clusters))

            for i, (label, wordcloud_text) in enumerate(self.wordcloud_data.items()):
                ax = axs[i] if num_clusters > 1 else axs

                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white'
                ).generate(wordcloud_text)

                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')

            if save:
                plt.savefig("./output/wordclould.png")

            if show:
                plt.show()
            else:
                plt.close()

        else:
            warnings.warn("self.metadata is not set. Please set self.metadata.", UserWarning)

    def cluster(self, min_clusters=2, max_clusters=10, plot=False, save=False, show=False, **kwargs):
        """
        Cluster the MSA data and obtain cluster labels.

        Parameters:
            min_clusters (int, optional): Minimum number of clusters (default is 2).
            max_clusters (int, optional): Maximum number of clusters (default is 10).
            plot (bool, optional): If True, the clustering results will be plotted (default is False).
            save (bool, optional): If True and `plot` is also True, the plotted results will be saved to a file.
            show (bool, optional): If True and `plot` is also True, the plotted results will be displayed.
            **kwargs: Additional keyword arguments passed to the clustering method.

        Notes:
        - This method performs clustering on the MSA data and assigns cluster labels to sequences.
        - Clustering is done using the single-linkage method.
        - The optimal number of clusters is determined using silhouette scores.
        - The cluster labels are stored in the 'labels' attribute of the MSA object.

        Example:
        msa = MSA('example.fasta')
        msa.map_positions()
        msa.cleanse()
        msa.cluster(min_clusters=3, plot=True, save=True)
        """
        # Check attribute dependencies and run the dependent method if needed
        if not hasattr(self, 'coordinates') or self.coordinates is None:
            self.reduce()  # Assuming this is the method that populates self.coordinates

        # Define a range of potential numbers of clusters to evaluate
        k_values = range(min_clusters, max_clusters + 1)
        silhouette_scores = []

        # Calculate the linkage once
        self.tree = fastcluster.linkage(self.coordinates, method='ward')

        # Perform clustering for different numbers of clusters and compute silhouette scores
        for k in k_values:
            # Calculate silhouette score for each value of k
            labels = fcluster(self.tree, k, criterion='maxclust')
            silhouette_scores.append(silhouette_score(self.coordinates, labels))

        # Find the index of the maximum silhouette score
        best_index = np.argmax(silhouette_scores)

        # Get the best value of k and get the cluster labels for each sequence in the MSA
        best_k = k_values[best_index]
        self.labels = fcluster(self.tree, best_k, criterion='maxclust')

        if plot:
            # self._plot_dendrogram(save=save, show=show)
            self._plot_wordclouds(save=save, show=show, **kwargs)

    def _plot_clustermap(self, processed=False, save=False, show=False):
        """
        Generate and display a clustermap of the encoded data using seaborn.

        Parameters:
            processed (bool, optional): If True, generates the 'after' clustermap using selected features.
                                        If False, generates the 'before' clustermap using all features.
            save (bool, optional): Whether to save the generated clustermap (default is False).
            show (bool, optional): Whether to display the generated clustermap (default is False).
        """
        data = self.encoded[self.selected_features] if processed else self.encoded

        # # Upscale the data
        # upscaled = zoom(data, zoom=2, order=1)  # Here, 3 is the zoom factor, adjust as needed

        params = {
            "method": "average", "cmap": "viridis"  # , "figsize": (6, 6)
        }

        if processed:
            # 'After' cluster map using precomputed dendrogram
            params.update({"row_linkage": self.tree})

        # g = sns.clustermap(upscaled, **params)
        g = sns.clustermap(data, **params)

        # Store clustered order
        self.sequence_order = g.dendrogram_row.reordered_ind
        self.residue_order = g.dendrogram_col.reordered_ind

        # Hide xticks and yticks for 'After' cluster map
        g.ax_heatmap.set_xticks([])
        g.ax_heatmap.set_yticks([])

        save_path = f"./output/clustermap_{'after' if processed else 'before'}.png"

        if save:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()

    def encode(self, plot=False, save=False, show=False):
        """
        One-hot encodes the MSA data and optionally plots a clustermap of the encoded data.

        This method encodes the MSA data using one-hot encoding. It then stores
        the encoded data frame in the 'encoded' attribute of the object. If desired,
        the method can also generate and optionally save a clustermap visualization
        of the encoded data before any feature selection is performed.

        Parameters:
            plot (bool, optional): Whether to plot the generated clustermap (default is False).
            save (bool, optional): Whether to save the generated clustermap (default is False).
            show (bool, optional): Whether to display the generated clustermap (default is False).
        """
        # Check attribute dependencies and run the dependent method if needed
        if not hasattr(self, 'labels') or self.labels is None:
            self.cluster()  # Assuming this is the method that populates self.labels

        # Extract X (features)
        x = self.unique

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
        self.encoded = pd.DataFrame.sparse.from_spmatrix(x_encoded, columns=encoded_feature_names)

        if plot:
            # Generate the 'before' clustermap
            self._plot_clustermap(processed=False, save=save, show=show)

    def _plot_pareto(self, save=False, show=True):
        """
        Generate and display a Pareto plot of the feature importance.

        Parameters:
            save (bool, optional): If True, the plotted results will be saved to a file (default is False).
            show (bool, optional): If True, the plotted results will be displayed (default is True).

        Notes:
        - This method produces a Pareto plot that visualizes the distribution of feature importance.
        - This plot can be useful in understanding which features contribute the most to model performance.
        """
        sorted_features = self.sorted_importance.index

        # Create a figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(16, 4))

        # Bar chart of percentage importance using a color near the beginning of the colormap
        xvalues = range(len(sorted_features))
        ax1.bar(xvalues, self.sorted_importance, color=my_palette(0.2))
        ax1.set_ylabel('Summed Importance', fontsize=16)
        ax1.tick_params(axis='y', labelsize=12)

        # Create a second y-axis for the line chart
        ax2 = ax1.twinx()

        # Line chart of cumulative percentage importance using a color near the end of the colormap
        ax2.plot(xvalues, np.cumsum(self.sorted_importance) / np.sum(self.sorted_importance), color=my_palette(0.8),
                 marker='.')
        ax2.set_ylabel('Cumulative Importance', fontsize=16)
        ax2.tick_params(axis='y', labelsize=12)

        # Rotate x-axis labels for better visibility
        plt.xticks(xvalues, sorted_features)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)

        if save:
            plt.savefig("./output/pareto_chart.png")

        if show:
            plt.show()
        else:
            plt.close()

    def select_features(self, n_estimators=None, random_state=None, plot=False, save=False, show=False):
        """
        Select important features (residues) from the encoded MSA data using RandomForest.

        Parameters:
            n_estimators (int, optional): Parameter n_estimators for RandomForest.
            random_state: (int, optional): Parameter random_state for RandomForest.
            plot (bool, optional): Whether to plot feature selection results (default is False).
            save (bool, optional): Whether to save feature selection results (default is False).
            show (bool, optional): Whether to show feature selection results (default is False).
        """
        # Check if the data has been encoded
        if not hasattr(self, 'encoded'):
            self.encode()

        # Extract y (labels)
        y = pd.get_dummies(self.labels).astype(int) if len(np.unique(self.labels)) > 2 else self.labels

        # Create and train the Random Forest classifier
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        rf.fit(self.encoded, y)

        # Feature selection
        feature_selector = SelectFromModel(rf, threshold='median')
        feature_selector.fit_transform(self.encoded, y)
        selected_feature_indices = feature_selector.get_support(indices=True)
        selected_features = self.encoded.columns[selected_feature_indices]

        # Calculate feature importance for original columns
        sorted_importance = pd.DataFrame(
            {
                'Residues': selected_features,
                'Importance': rf.feature_importances_[selected_feature_indices],
                'Columns': map(lambda ftr: int(ftr.split('_')[0]), selected_features)
            }
        )[['Columns', 'Importance']].groupby('Columns').sum()['Importance'].sort_values(ascending=False)

        # Store the selected features and their importance
        self.selected_features = selected_features
        self.sorted_importance = sorted_importance

        if plot:
            self._plot_clustermap(processed=True, save=save, show=show)
            self._plot_pareto(save=save, show=show)

    def _plot_logos(self, color_scheme='NajafabadiEtAl2017', save=False, show=False, **kwargs):
        """
        Generate and plot logos from the MSA data for each cluster label.

        This method generates and plots the logos using the dmslogo library.
        It plots the logos for each cluster label and displays them in a single figure.

        Parameters:
            color_scheme (str, optional): The color scheme for plotting (default is 'NajafabadiEtAl2017').
            plot (bool, optional): Whether to plot the generated logos (default is False).
            save (bool, optional): Whether to save the generated logos (default is False).
            show (bool, optional): Whether to show the generated logos (default is False).
        """
        print("Executing _plot_logos...")
        self.logos_data = {}
        unique_labels = np.unique(self.labels)

        for label in unique_labels:
            sub_msa = self.unique[sorted(self.selected_columns)].iloc[self.labels == label]
            data = sub_msa.T.apply(lambda col: col.value_counts(normalize=True), axis=1).fillna(0)
            self.logos_data[label] = data

        color_schemes = lm.list_color_schemes()
        color_schemes_list = sorted(
            color_schemes.loc[color_schemes.characters == 'ACDEFGHIKLMNPQRSTVWY'].color_scheme.values
        )

        if color_scheme not in color_schemes_list:
            raise ValueError(f"color scheme must be in {color_schemes_list}")

        n_labels = len(unique_labels)
        fig, axs = plt.subplots(nrows=n_labels, ncols=1, figsize=(8, 4 * n_labels), sharex=True)
        axs = np.array(axs, ndmin=1)

        for i, (label, data) in enumerate(self.logos_data.items()):
            msa_columns = data.index.tolist()
            data = data.reset_index(drop=True)
            ax = axs[i]
            seq_logo = lm.Logo(data, ax=ax, color_scheme=color_scheme, vpad=.1, width=.8)
            seq_logo.style_spines(visible=False)
            ax.set_xticks(range(len(msa_columns)))
            ax.set_xticklabels(msa_columns, fontsize=24)
            ax.tick_params(axis='y', labelsize=12)  # Increase y-ticks fontsize

        plt.tight_layout()
        save_path = f"./output/sdp_combined_logo.png"
        print(f"Saving to: {save_path}")

        if save:
            plt.savefig(f"./output/sdp_combined_logo.png")
            print("Logo saved successfully!")

        if show:
            plt.show()
        else:
            plt.close()

    def select_residues(self, threshold=0.9, top_n=None, plot=False, save=False, show=True, **kwargs):
        """
        Select and store residues that are candidates for Specificity-determining Positions (SDPs) from the MSA data.

        Parameters:
            threshold (float, optional): The threshold for selecting residues based on their importance. Default is 0.9.
            top_n (int, optional): Selects the top N residues based on their importance. If set to None, selection is
            based on the threshold. Default is None.
            plot (bool, optional): If True, visual representation of the selected residues will be plotted. Default is
            False.
            save (bool, optional): If True, the plot will be saved to a file. This parameter is considered only if
            `plot` is True. Default is False.
            show (bool, optional): If True, the plot will be displayed. This parameter is considered only if `plot` is
            True. Default is True.

        Notes:
        - This method selects residues that are candidates for Specificity-determining Positions (SDPs) based on their
        importance.
        - The importance can be determined through various methods (e.g., feature importance from a classifier,
        conservation score).
        - Selected residues can offer insights into the functional or structural significance in the protein family.
        """
        # Check attribute dependencies and run the dependent method if needed
        if not hasattr(self, 'selected_features') or self.selected_features is None or not hasattr(self,
                                                                                                   'sorted_importance') or self.sorted_importance is None:
            self.select_features(plot=False, save=False, show=False,
                                 **kwargs)  # Assuming this is the method that populates self.sorted_importance

        # Calculate cumulative sum of importance
        cumulative_importance = np.cumsum(self.sorted_importance) / np.sum(self.sorted_importance)
        # Find the index where cumulative importance exceeds or equals a threshold (default is 0.9)
        cutoff = np.where(cumulative_importance >= threshold)[0][0]
        # Get the values from sorted_features up to the index
        selected_columns = self.sorted_importance.index[:cutoff + 1].values
        # Filter the selected features to get the 'top_n' most important residues
        self.selected_columns = selected_columns if top_n is None else selected_columns[:top_n]
        # Create an empty DataFrame with columns
        self.profiles = pd.DataFrame(columns=sorted(self.selected_columns))
        # Iterate through rows of msa
        for header, row in self[sorted(self.selected_columns)].iterrows():
            pos_map = self.positions_map.get(header, {})
            series = pd.Series(
                {col: f"{seq3(aa)}{pos_map.get(col, '?')}" for col, aa in row.items()}
            )
            self.profiles = pd.concat([self.profiles, series.to_frame().T], axis=0)
        self.profiles.index = self.index

        if plot:
            self._plot_perceptual_map(processed=True, save=save, show=show)
            self._plot_logos(save=save, show=show, **kwargs)


def main():
    """
    Main function to process the Multiple Sequence Alignment (MSA) data.
    """
    # Define command-line arguments
    parser = argparse.ArgumentParser(
        description="Process Multiple Sequence Alignment (MSA) data."
    )

    # Custom usage message
    usage = "python protlearn.py MSA_FILE [OPTIONS]"
    parser.usage = usage

    # Define command-line arguments
    parser.add_argument(
        "data", type=str, help="Path to the MSA file"
    )
    parser.add_argument(
        "--metadata", type=str, default=None,
        help="Path to the metadata file in tsv format. Optional."
    )
    parser.add_argument(
        "--hide",
        action="store_true",
        help="Hide the output plots. By default, outputs are shown."
    )
    parser.add_argument(
        '--export',
        action="store_true",
        help="Whether to export the MSA object as a pickle file. Optional."
    )

    args = parser.parse_args()

    try:
        # Creates output directory if not exists
        if not os.path.exists('./output'):
            os.makedirs('./output')

        # Downloading nltk resources is only necessary for _plot_weblog, which requires metadata
        if args.metadata:
            download_nltk_resources()

        # Initializes MSA object
        msa = MSA(args.data, metadata_file=args.metadata or None)
        viz_params = {
            'plot': True,
            'save': True,
            'show': not args.hide
        }
        msa.cleanse(**viz_params)
        msa.reduce(n_components=3, **viz_params)
        msa.cluster(**viz_params)
        msa.select_features(n_estimators=1000, random_state=42, **viz_params)
        msa.select_residues(top_n=3, **viz_params)

        if args.export:
            filename = os.path.basename(args.data)
            basename, _ = os.path.splitext(filename)
            with open(f"./output/{basename}.pkl", "wb") as f:
                pickle.dump(msa, f)

        return True

    except Exception as e:
        raise


if __name__ == "__main__":
    try:
        sys.exit(0 if main() else 1)
    except Exception as e:
        import traceback

        traceback.print_exc()  # This will print the full traceback
        print(f"An error occurred: {type(e).__name__} - {e}")
        sys.exit(2)
