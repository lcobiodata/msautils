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
from wordcloud import WordCloud
import nltk
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import re
from prince import MCA
from Bio.SeqUtils import seq3
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import fastcluster
from scipy.cluster.hierarchy import fcluster
from collections import defaultdict
import logomaker as lm
import sys
import argparse

# Download necessary resources from NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


class MSA(pd.DataFrame):
    """
    A class for processing and analyzing Multiple Sequence Alignments (MSA).
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the MSA object.

        Args:
            *args: Variable-length positional arguments.
            **kwargs: Variable-length keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.msa_file = None  # Path to the MSA file
        self.raw_data = None  # Raw data from the MSA
        self.positions_map = None  # Mapping of positions in the MSA
        self.dirty = None  # Dirty data (before cleansing)
        self.clean = None  # Clean data (after cleansing)
        self.data = None  # Processed MSA data
        self.alphabet = None  # Alphabet used in the MSA
        self.analysis = None  # Analysis results
        self.coordinates = None  # Reduced data coordinates
        self.labels = None  # Sequence labels or clusters
        self.sorted_importance = None  # Sorted feature importances
        self.selected_features = None  # Selected features
        self.selected_columns = None  # Selected columns from the data
        self.profiles = None  # Residue profiles
        # self.clusters = None  # Clusters (uncomment if needed)
        self.wordcloud_data = None  # Word cloud data
        self.logos_data = None  # Logo data
        self.plots = None  # Plots and visualizations

    def parse_msa_file(self, msa_file, msa_format="fasta", *args, **kwargs):
        """
        Parse an MSA file and store the raw data in the MSA object.

        Args:
            msa_file (str): The path to the MSA file.
            msa_format (str, optional): The format of the MSA file (default is "fasta").
            *args: Additional positional arguments to pass to SeqIO.parse.
            **kwargs: Additional keyword arguments to pass to SeqIO.parse.
        """
        self.msa_file = msa_file  # Store the path to the MSA file
        headers, sequences = [], []

        # Parse the MSA file and extract headers and sequences
        for record in SeqIO.parse(self.msa_file, msa_format, *args, **kwargs):
            headers.append(record.id)
            sequences.append(record.seq)

        # Create a DataFrame to store the raw MSA data with sequences as rows and headers as index
        self.raw_data = pd.DataFrame(np.array(sequences), index=headers)

        # Initialize a dictionary to store plots and visualizations
        self.plots = defaultdict(list)

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
            'Seq1/1-100': {'A': 1, 'C': 2, ...},
            'Seq2/101-200': {'A': 101, 'C': 102, ...},
            ...
        }

        This mapping is useful for downstream analysis that requires knowing the position
        of residues in the MSA.

        Note:
        - Residues represented by '-' (indels/gaps) are not included in the mapping.

        Example:
        msa = MSA()
        msa.parse_msa_file('example.fasta')
        msa.map_positions()

        Access the mapping:
        positions = msa.positions_map
        """
        # Initialize a dictionary to store residue positions
        self.positions_map = defaultdict(dict)

        # Iterate through sequence headers and sequences
        for header in self.raw_data.index:
            sequence = self.raw_data.loc[header]

            # Extract the starting position from the header
            offset = int(header.split('/')[1].split('-')[0])
            count = offset - 1

            # Iterate through residues in the sequence
            for index, value in zip(sequence.index, sequence.values):
                if value != '-':
                    count += 1
                    # Store the residue position in the positions_map dictionary
                    self.positions_map[header][index] = count

    def cleanse_data(self, indel='-', remove_lowercase=True, threshold=.9, plot=False):
        """
        Cleanse the MSA data by removing columns and rows with missing values.

        Parameters:
            indel (str, optional): The character representing gaps/indels (default is '-').
            remove_lowercase (bool, optional): Whether to remove lowercase characters (default is True).
            threshold (float, optional): The threshold for missing values (default is 0.9).
            plot (bool, optional): Whether to plot a heatmap of missing values (default is False).
        """
        # Define characters to remove based on parameters
        to_remove = [indel]

        # Create a copy of the raw data as the 'dirty' data
        self.dirty = self.raw_data.copy()

        # Check if lowercase characters should be removed
        if remove_lowercase:
            to_remove += [chr(i) for i in range(ord('a'), ord('z') + 1)]
        else:
            self.dirty = self.raw_data.applymap(str.upper).copy()

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
        self.data = self.clean.reset_index(drop=True) \
            .drop_duplicates() \
            .fillna('-') \
            .copy()

        # Store a flag indicating whether plotting is enabled
        self.plots['cleanse_data'] = plot

        # If plotting is enabled, plot heatmaps
        if plot:
            self._plot_cleanse_heatmaps()

    def _plot_cleanse_heatmaps(self):
        """
        Generate and display cleansing heatmaps plot on the specified axes.
        """
        # Create a figure with two subplots (heatmaps)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Create the heatmap before cleansing on the first Axes
        ax1 = axes[0]  # First subplot
        heatmap_before = ax1.imshow(self.dirty.isna().astype(int), cmap='binary', aspect='auto', extent=[0, 1, 0, 1])

        # Create the heatmap after cleansing on the second Axes
        ax2 = axes[1]  # Second subplot
        heatmap_after = ax2.imshow(self.clean.isna().astype(int), cmap='binary', aspect='auto', extent=[0, 1, 0, 1])

        # Create a shared color bar axis
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust the position and size as needed

        # Add color bars to the right of the heatmaps with the shared color bar axis
        cbar_before = plt.colorbar(heatmap_before, cax=cax)
        cbar_before.set_label('Missing Values (NaN)')
        cbar_after = plt.colorbar(heatmap_after, cax=cax)  # Use the same color bar axis
        cbar_after.set_label('Missing Values (NaN)')

        ax1.axis('off')  # Turn off axis labels for the first subplot
        ax2.axis('off')  # Turn off axis labels for the second subplot

        plt.show()

    def reduce(self, plot=False, *args, **kwargs):
        """
        Perform Multidimensional Correspondence Analysis (MCA) on the MSA data to reduce dimensionality.

        Parameters:
            plot (bool, optional): Whether to plot the results (default is False).
            *args, **kwargs: Additional arguments and keyword arguments for the MCA.

        Notes:
        - Multidimensional Correspondence Analysis (MCA) is used to reduce the dimensionality of the MSA data.
        - The MCA results are stored in the 'analysis' attribute of the MSA object.
        - The row coordinates after reduction are stored in the 'coordinates' attribute.

        Example:
        msa = MSA()
        msa.parse_msa_file('example.fasta')
        msa.map_positions()
        msa.cleanse_data()
        msa.reduce(plot=True)
        """
        # Perform MCA
        mca = MCA(*args, **kwargs)
        mca.fit(self.data)
        self.analysis = mca

        # Get the row coordinates
        self.coordinates = self.analysis.transform(self.data)

        self.plots['analyse'] = plot  # Set a flag to indicate plotting
        if plot:
            self._plot_mca()

    def _plot_mca(self):
        """
        Plot the results of Multidimensional Correspondence Analysis (MCA).

        Notes:
        - This method plots a scatter plot of both sequences and residues overlaid based on the MCA results.

        Example:
        msa = MSA()
        msa.parse_msa_file('example.fasta')
        msa.map_positions()
        msa.cleanse_data()
        msa.reduce(plot=True)
        """
        try:
            # Plot together the scatter plot of both sequences and residues overlaid
            self.analysis.plot(
                self.data,
                x_component=0,
                y_component=1
            )
        finally:
            pass

    def label_sequences(self, min_clusters=2, max_clusters=10, method='single-linkage', plot=False):
        """
        Cluster the MSA data and obtain cluster labels.

        Parameters:
            min_clusters (int, optional): Minimum number of clusters (default is 2).
            max_clusters (int, optional): Maximum number of clusters (default is 10).
            method (str, optional): Clustering method ('k-means' or 'single-linkage') (default is 'single-linkage').
            plot (bool, optional): Whether to plot the clustering results (default is False).

        Notes:
        - This method performs clustering on the MSA data and assigns cluster labels to sequences.
        - Clustering can be done using either k-means or single-linkage methods.
        - The optimal number of clusters is determined using silhouette scores.
        - The cluster labels are stored in the 'labels' attribute of the MSA object.

        Example:
        msa = MSA()
        msa.parse_msa_file('example.fasta')
        msa.map_positions()
        msa.cleanse_data()
        msa.label_sequences(method='single-linkage', min_clusters=3, plot=True)
        """
        if method not in ['k-means', 'single-linkage']:
            raise ValueError("method must be 'k-means' or 'single-linkage")

        coordinates = np.array(self.coordinates)
        # Define a range of potential numbers of clusters to evaluate
        k_values = range(min_clusters, max_clusters)
        # Perform clustering for different numbers of clusters and compute silhouette scores
        model, silhouette_scores = None, []
        for k in range(min_clusters, max_clusters + 1):
            if method == 'k-means':
                model = KMeans(n_clusters=k, n_init=max_clusters)  # Set n_init explicitly
                model.fit(coordinates)
                labels = model.labels_
                score = silhouette_score(coordinates, labels)
                silhouette_scores.append(score)
            elif method == 'single-linkage':
                # Run clustermap with potential performance improvement
                model = fastcluster.linkage(coordinates, method='ward')
                # Calculate silhouette score for each value of k
                labels = fcluster(model, k, criterion='maxclust')
                silhouette_scores.append(silhouette_score(coordinates, labels))

        # Find the index of the maximum silhouette score
        best_index = np.argmax(silhouette_scores)
        # Perform the actual modeling depending on the chosen algorithm
        if method == 'k-means':
            # Find the best number of clusters based on the highest silhouette score
            best_num_clusters = best_index + min_clusters
            # Perform clustering with the best number of clusters
            model = KMeans(n_clusters=best_num_clusters, n_init=max_clusters)
            model.fit(coordinates)
            self.labels = model.labels_
        elif method == 'single-linkage':
            # Get the best value of k
            best_k = k_values[best_index]
            # Get the cluster labels for each sequence in the MSA
            self.labels = fcluster(model, best_k, criterion='maxclust')

        self.plots['get_labels'] = plot  # Set a flag to indicate plotting
        if plot:
            self._plot_sequence_labels()

    def _plot_sequence_labels(self):
        """
        Generate and display a cluster labels plot on the specified axes.

        Notes:
        - This method generates a scatter plot of sequences based on MCA coordinates, with points colored by cluster labels.
        - It provides a visualization of the clustering results.

        Example:
        msa = MSA()
        msa.parse_msa_file('example.fasta')
        msa.map_positions()
        msa.cleanse_data()
        msa.label_sequences(method='single-linkage', min_clusters=3, plot=True)
        """
        fig, ax = plt.subplots(figsize=(8, 6))  # Create a single subplot
        coordinates = np.array(self.coordinates)
        ax.scatter(coordinates[:, 0], coordinates[:, 1], c=self.labels, cmap='viridis', alpha=0.5)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        # ax.set_title("Scatter Plot of Sequences Clusters out of MCA Coordinates")
        plt.show()

    def generate_wordclouds(self, path_to_metadata=None, column='Protein names', plot=False):
        """
        Generate word cloud visualizations from protein names in a DataFrame.

        Parameters:
            path_to_metadata (str, default=None): Path to metadata file in tsv format.
            column (str, default='Protein names'): The name of the column in the DataFrame containing protein names.
            plot (bool, default=False): Whether to plot word clouds.

        This method extracts substrate and enzyme names from the specified column using regular expressions,
        normalizes the names, and creates word cloud plots for each cluster of sequences.

        Example:
        msa = MSA()
        msa.parse_msa_file('example.fasta')
        msa.map_positions()
        msa.cleanse_data()
        msa.label_sequences(method='single-linkage', min_clusters=3)
        msa.generate_wordclouds(path_to_metadata='metadata.tsv', plot=True)
        """
        # Read the TSV file into a DataFrame
        if path_to_metadata is not None:
            metadata = pd.read_csv(path_to_metadata, delimiter='\t')
            self.wordcloud_data = {}

            # Process data for each cluster
            for label in set(self.labels):
                indices = self.data.iloc[self.labels == label].index
                headers = self.raw_data.index[indices]
                entry_names = [header.split('/')[0] for header in headers]

                # Perform a left join using different key column names
                result = metadata[metadata['Entry Name'].isin(entry_names)].copy()

                # Extract the substrate and enzyme names using regular expressions
                matches = result[column].str.extract(r'(.+?) ([\w\-,]+ase)', flags=re.IGNORECASE)

                # String normalization pipeline
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

                # Combine normalized labels and create word cloud text
                wordcloud_text = ' '.join(
                    sorted(
                        set([string for string in result.Label.values.tolist() if len(string) > 0])
                    )
                )
                self.wordcloud_data[label] = wordcloud_text

            self.plots['generate_wordcloud'] = plot  # Set a flag to indicate plotting
            if plot:
                self._plot_wordclouds()

    def _plot_wordclouds(self):
        """
        Plot word clouds generated by the generate_wordclouds method.

        This method plots the word clouds generated by the generate_wordclouds method using the
        wordcloud library. It plots the word clouds for each cluster label and displays them
        in a single figure.

        This method is intended for internal use and should not be called directly.
        """
        # Create a figure and subplots for word clouds
        num_clusters = len(self.wordcloud_data)
        fig, axs = plt.subplots(num_clusters, 1, figsize=(10, 6 * num_clusters))

        for i, (label, wordcloud_text) in enumerate(self.wordcloud_data.items()):
            ax = axs[i] if num_clusters > 1 else axs  # Use a single subplot if there's only one cluster

            # Generate the word cloud for the current cluster
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white'
            ).generate(wordcloud_text)

            # Plot the word cloud on the current subplot
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            # ax.set_title(f'Wordcloud of Protein Names for Cluster {label}')

        # Show the plot
        plt.show()

    def select_features(self, n_estimators=None, random_state=None, plot=False):
        """
        Select important features (residues) from the MSA data.

        Parameters:
            n_estimators (int, optional): Parameter n_estimators for RandomForest.
            random_state: (int, optional): Parameter random_state for RandomForest.
            plot (bool, optional): Whether to plot feature selection results (default is False).
        """
        # Extract X (features) and y (labels)
        x = self.data
        y = pd.get_dummies(self.labels).astype(int) if len(set(self.labels)) > 2 else self.labels

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

        # Store the selected features and their importances
        self.selected_features = selected_features
        self.sorted_importance = sorted_importance

        self.plots['select_features'] = plot  # Set a flag to indicate plotting
        if plot:
            self._plot_pareto()

    def _plot_pareto(self):
        """
        Generate and display feature selection plot on the specified axes.
        """
        sorted_features = self.sorted_importance.index

        # Create a figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(16, 4))

        # Bar chart of percentage importance
        xvalues = range(len(sorted_features))
        ax1.bar(xvalues, self.sorted_importance, color='b')
        ax1.set_ylabel('Summed Importance', fontsize=16)
        ax1.tick_params(axis='y', labelsize=12)

        # Create a second y-axis for the line chart
        ax2 = ax1.twinx()

        # Line chart of cumulative percentage importance
        ax2.plot(xvalues, np.cumsum(self.sorted_importance) / np.sum(self.sorted_importance), color='r', marker='.')
        ax2.set_ylabel('Cumulative Importance', fontsize=16)
        ax2.tick_params(axis='y', labelsize=12)

        # Rotate x-axis labels for better visibility
        plt.xticks(xvalues, sorted_features)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)

        # Show the plot
        plt.show()

    def select_residues(self, threshold=0.9, top_n=None, plot=False):
        """
        Select and store residues to be candidates for Specificity-determining Positions (SDPs) from the MSA data.

        Parameters:
            threshold (float, optional): The threshold for selecting residues based on importance (default is 0.9).
            top_n (int, optional): The top N residues to select based on importance (default is None).
            plot (bool, optional): Whether to plot the selected residues (default is False).
        """
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
        # Iterate through rows of msa.data
        for index, row in self.data[self.selected_columns].iterrows():
            header = self.raw_data.index[index]
            series = pd.Series(
                {col: f"{seq3(aa)}{self.positions_map[header].get(col, '?')}" for col, aa in row.items()}
            ).sort_index()
            self.profiles = pd.concat([self.profiles, series.to_frame().T], axis=0)
        # Set the custom index to the resulting DataFrame
        self.profiles.index = self.raw_data.index[self.data.index]

        self.plots['get_sdps'] = plot  # Set a flag to indicate plotting
        if plot:
            self._plot_selected_residues()

    def _plot_selected_residues(self):
        """
        Generate and display selected residues plot on the specified axes.
        """
        selected_residues = [
            feature for feature in self.selected_features if int(feature.split('_')[0]) in self.selected_columns
        ]
        df_res = self.analysis.column_coordinates(self.data[self.selected_columns]).loc[selected_residues]
        residues = []
        for res_idx in df_res.index:
            msa_column, amino_acid = res_idx.split('_')
            three_letter_aa = seq3(amino_acid)
            residues.append(three_letter_aa + msa_column)
        df_res = df_res.set_index(pd.Index(residues))
        # Create figure
        plt.figure(figsize=(8, 6))
        # Scatter plot of sequences colored by cluster labels
        seq_coord = np.array(self.coordinates)
        plt.scatter(
            seq_coord[:, 0],
            seq_coord[:, 1],
            c=self.labels,
            cmap='viridis',
            alpha=0.5,
            label='Clustered Sequence'
        )
        # Scatter plot of labeled residues
        plt.scatter(df_res[0], df_res[1], marker='x', color='black', s=50, label='Selected Residues')
        for i, (x, y) in enumerate(zip(df_res[0], df_res[1])):
            plt.annotate(df_res.index[i], (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
        # Set labels and title
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        # plt.title('Conceptual Map of Clustered Sequences and Selected Residues')
        # Show the plot
        plt.legend()
        plt.grid()

    def generate_logos(self, plot=False):
        """
        Generate logos from the MSA data for each cluster label.

        Parameters:
            plot (bool, optional): Whether to plot the generated logos (default is False).
        """
        if plot:
            self.plots['generate_logos'] = plot  # Set a flag to indicate plotting
            self.logos_data = {}  # Initialize logos_data as an empty dictionary
            for label in set(self.labels):
                sub_msa = self.data[sorted(self.selected_columns)].iloc[self.labels == label]

                # Calculate sequence frequencies for each position
                data = sub_msa.T.apply(lambda col: col.value_counts(normalize=True), axis=1).fillna(0)

                # Store the seq_logo in logos_data with the label as the key
                self.logos_data[label] = data
            self._plot_logos()

    def _plot_logos(self, color_scheme='NajafabadiEtAl2017'):
        """
        Plot logos generated by the generate_logos method.

        This method plots the logos generated by the generate_logos method using the
        dmslogo library. It plots the logos for each cluster label and displays them
        in a single figure.

        Parameters:
            color_scheme (str, optional): The color scheme for plotting (default is 'NajafabadiEtAl2017').

        This method is intended for internal use and should not be called directly.
        """
        color_schemes = lm.list_color_schemes()
        color_schemes_list = sorted(
            color_schemes.loc[color_schemes.characters == 'ACDEFGHIKLMNPQRSTVWY'].color_scheme.values)
        if color_scheme not in color_schemes_list:
            raise ValueError(f"color scheme must be in {color_schemes_list}")
        # Create a figure and subplots for logos
        num_clusters = len(self.logos_data)

        for i, (label, data) in enumerate(self.logos_data.items()):
            msa_columns = data.index.tolist()

            data = data.reset_index(drop=True)

            # Create a sequence logo from the DataFrame
            seq_logo = lm.Logo(data,
                               color_scheme=color_scheme,
                               vpad=.1,
                               width=.8)

            # Customize the appearance of the logo
            seq_logo.style_spines(visible=False)

            # Get the axes from the logo object
            ax = seq_logo.ax

            # Customize subplot labels and title
            ax.set_xticks(range(len(msa_columns)))
            ax.set_xticklabels(msa_columns, fontsize=24)

        # Show the plot
        plt.show()


def main(args):
    """
    Main function to process the Multiple Sequence Alignment (MSA) data.

    Args:
        args: Command-line arguments parsed by argparse.

    This function performs a series of data processing and analysis steps on the MSA data,
    including parsing, cleansing, dimension reduction, clustering, word cloud generation,
    feature selection, selecting residues, and generating logos.
    """
    # Create data frame from raw data and clean it
    msa_file = args.msa_file
    msa = MSA()
    msa.parse_msa_file(msa_file)
    msa.map_positions()
    msa.cleanse_data(plot=args.plot_cleanse)
    msa.reduce(plot=args.plot_reduce)
    msa.label_sequences(method='single-linkage', min_clusters=3, plot=args.plot_label)
    msa.generate_wordclouds(path_to_metadata='pf00848-metadata.tsv', plot=args.plot_wordclouds)
    msa.select_features(n_estimators=1000, random_state=42, plot=args.plot_select_features)
    msa.select_residues(top_n=3, plot=args.plot_select_residues)
    msa.generate_logos(plot=args.plot_logos)


def parse_args():
    """
    Parse command-line arguments using argparse.

    Returns:
        Namespace: An object containing parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process Multiple Sequence Alignment (MSA) data.")
    parser.add_argument("msa_file", type=str, help="Path to the MSA file")
    parser.add_argument("--plot-cleanse", action="store_true", help="Plot cleanse data")
    parser.add_argument("--plot-reduce", action="store_true", help="Plot data reduction")
    parser.add_argument("--plot-label", action="store_true", help="Plot sequence labels")
    parser.add_argument("--plot-wordclouds", action="store_true", help="Plot word clouds")
    parser.add_argument("--plot-select-features", action="store_true", help="Plot feature selection")
    parser.add_argument("--plot-select-residues", action="store_true", help="Plot selected residues")
    parser.add_argument("--plot-logos", action="store_true", help="Plot logos")

    # Create a custom usage message
    usage = "python your_script.py MSA_FILE [OPTIONS]"
    parser.usage = usage

    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        main(args)
    except Exception as e:
        print(f"An error occurred: {type(e).__name__} - {e}")
        sys.exit(1)
