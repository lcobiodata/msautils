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
        """
        super().__init__(*args, **kwargs)
        self.msa_file = None
        self.raw_data = None
        self.positions_map = None
        self.dirty = None
        self.clean = None
        self.data = None
        self.alphabet = None
        self.analysis = None
        self.coordinates = None
        self.labels = None
        self.sorted_importance = None
        self.selected_features = None
        self.selected_columns = None
        self.profiles = None
        # self.clusters = None
        self.wordcloud_data = None
        self.logos_data = None
        self.plots = None

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
        self.raw_data = pd.DataFrame(np.array(sequences), index=headers)
        self.plots = defaultdict(list)

    def map_positions(self):
        """
        Map residue positions in the Multiple Sequence Alignment (MSA) based on sequence headers.

        This method calculates the position of each residue in the MSA sequences and stores
        the mapping in the 'positions_map' attribute of the MSA object. The mapping is
        based on the sequence headers, which typically include information about the
        sequence's starting position.

        The 'positions_map' is a dictionary with sequence headers as keys and a sub-dictionary
        as values. The sub-dictionary contains residue names (keys) and their corresponding
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
        self.positions_map = defaultdict(dict)
        for header in self.raw_data.index:
            sequence = self.raw_data.loc[header]
            offset = int(header.split('/')[1].split('-')[0])
            count = offset - 1
            for index, value in zip(sequence.index, sequence.values):
                if value != '-':
                    count += 1
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
        to_remove = [indel]
        self.dirty = self.raw_data.copy()
        if remove_lowercase:
            to_remove += [chr(i) for i in range(ord('a'), ord('z') + 1)]
        else:
            dirty = self.raw_data.applymap(str.upper).copy()
        self.dirty.replace(to_remove, np.nan, inplace=True)
        min_rows = int(threshold * self.dirty.shape[0])
        # Remove columns with NaN values above the threshold
        self.clean = self.dirty.copy()
        self.clean.dropna(thresh=min_rows, axis=1, inplace=True)
        min_cols = int(threshold * self.clean.shape[1])
        # Remove rows with NaN values above the threshold
        self.clean.dropna(thresh=min_cols, axis=0, inplace=True)

        self.data = self.clean.reset_index(drop=True) \
            .drop_duplicates() \
            .fillna('-') \
            .copy()

        self.plots['cleanse_data'] = plot  # Set a flag to indicate plotting
        if plot:
            self._plot_cleanse_heatmaps()

    def _plot_cleanse_heatmaps(self):
        """
        Generate and display cleansing heatmaps plot on the specified axes.
            """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Create a figure with two subplots
        fig.suptitle("Cleansing Heatmaps", fontsize=16)

        # Create the heatmap before cleansing on the first Axes
        ax1 = axes[0]  # First subplot
        heatmap_before = ax1.imshow(self.dirty.isna().astype(int), cmap='binary', aspect='auto', extent=[0, 1, 0, 1])
        ax1.set_title('Before Cleansing')

        # Create the heatmap after cleansing on the second Axes
        ax2 = axes[1]  # Second subplot
        heatmap_after = ax2.imshow(self.clean.isna().astype(int), cmap='binary', aspect='auto', extent=[0, 1, 0, 1])
        ax2.set_title('After Cleansing')

        # Add color bars to the right of the heatmaps
        cbar_before = plt.colorbar(heatmap_before, ax=ax1)
        cbar_before.set_label('Missing Values (NaN)')
        cbar_after = plt.colorbar(heatmap_after, ax=ax2)
        cbar_after.set_label('Missing Values (NaN)')

        ax1.axis('off')  # Turn off axis labels for the first subplot
        ax2.axis('off')  # Turn off axis labels for the second subplot

        plt.show()

    def reduce(self, plot=False, *args, **kwargs):
        """
        Perform Multidimensional Correspondence Analysis (MCA) on the MSA data to reduce dimensionality.

        Parameters:
            plot (bool, optional): Whether to plot the results (default is False).
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
            plot (bool, optional): Whether to plot the clustering results (default is False).
            min_clusters (int, optional): Minimum amount of clusters.
            max_clusters (int, optional): Maximum amount of clusters.
            method (string, optional): Custering method (k-means or single-linkage).
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
        Generate and display cluster labels plot on the specified axes.
        """
        fig, ax = plt.subplots(figsize=(8, 6))  # Create a single subplot
        coordinates = np.array(self.coordinates)
        ax.scatter(coordinates[:, 0], coordinates[:, 1], c=self.labels, cmap='viridis', alpha=0.5)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title("Scatter Plot of Sequences Clusters out of MCA Coordinates")
        plt.show()

    def generate_wordclouds(self, path_to_metadata=None, column='Protein names', plot=False):
        """
        Generate a word cloud visualization from protein names in a DataFrame.

        Parameters:
            path_to_metadata (str, default=None): Path to metadata file in tsv format.
            column (str, default='Protein names'): The name of the column in the DataFrame containing protein names.
            plot (bool, default=False): Whether to plot output

        This function extracts substrate and enzyme names from the specified column using regular expressions,
        normalizes the names, and creates a word cloud plot.
        """
        # Read the TSV file into a DataFrame
        if path_to_metadata is not None:
            metadata = pd.read_csv(path_to_metadata, delimiter='\t')
            self.wordcloud_data = {}
            for label in set(self.labels):
                print("label")
                indices = self.data.iloc[self.labels == label].index
                print("indices")
                headers = self.raw_data.index[indices]
                print("headers")
                # Perform a left join using different key column names
                entry_names = [header.split('/')[0] for header in headers]
                print("entry_names")
                result = metadata[metadata['Entry Name'].isin(entry_names)].copy()
                print("result")
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
                print("result 2")
                wordcloud_text = ' '.join(
                    sorted(
                        set([string for string in result.Label.values.tolist() if len(string) > 0])
                    )
                )
                print(wordcloud_text)
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
            print(i, label, wordcloud_text)
            ax = axs[i] if num_clusters > 1 else axs  # Use a single subplot if there's only one cluster

            # Generate the word cloud for the current cluster
            wordcloud = WordCloud(
                width=80,
                height=40,
                background_color='white'
            ).generate(wordcloud_text)

            # Plot the word cloud on the current subplot
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'Wordcloud of Protein Names for Cluster {label}')

        # Adjust subplot layout
        plt.tight_layout()

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
        fig, ax1 = plt.subplots(figsize=(16, 4))
        # Bar chart of percentage importance
        xvalues = range(len(sorted_features))
        ax1.bar(xvalues, self.sorted_importance, color='b')
        ax1.set_ylabel('Summed Importance', fontsize=16)
        ax1.tick_params(axis='y', labelsize=12)
        # Line chart of cumulative percentage importance
        ax2 = ax1.twinx()
        ax2.plot(xvalues, np.cumsum(self.sorted_importance) / np.sum(self.sorted_importance), color='r', marker='.')
        ax2.set_ylabel('Cumulative Importance', fontsize=16)
        ax2.tick_params(axis='y', labelsize=12)
        # Rotate x-axis labels
        plt.xticks(xvalues, sorted_features)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
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
        plt.title('Conceptual Map of Clustered Sequences and Selected Residues')
        # Show the plot
        plt.legend()
        plt.grid()

    def generate_logos(self, plot=False):

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

        This method is intended for internal use and should not be called directly.
        """
        color_schemes = lm.list_color_schemes()
        color_schemes_list = sorted(
            color_schemes.loc[color_schemes.characters == 'ACDEFGHIKLMNPQRSTVWY'].color_scheme.values)
        if color_scheme not in color_schemes_list:
            raise ValueError(f"color scheme must be in {color_schemes_list}")
        # Create a figure and subplots for logos
        num_clusters = len(self.logos_data)
        fig, axs = plt.subplots(num_clusters, 1, figsize=(8, 2 * num_clusters), sharex=True)

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

            ax = axs[i] if num_clusters > 1 else axs  # Use a single subplot if there's only one cluster

            # Plot the sequence logo on the current subplot
            seq_logo.plot(ax=ax)

            # Customize subplot labels and title
            ax.set_xticks(range(len(msa_columns)))
            ax.set_xticklabels(msa_columns)
            ax.set_title(f'Cluster {label}')

        # Set common X-axis label (if needed)
        fig.text(0.5, 0.04, 'Residue Position', ha='center')

        # Adjust subplot layout
        plt.tight_layout()

        # Show the plot
        plt.show()

    # def generate_visualizations(self):
    #     """
    #     Generate and display visualizations for all deferred plots.
    #     """
    #     # Create a grid of subplots to display the visualizations
    #     num_plots = sum(flag for flag in self.plots.values() if flag)
    #     num_rows = num_plots // 2 + num_plots % 2  # Create rows for subplots
    #     num_cols = 2  # Two plots per row
    #
    #     fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 4 * num_rows))
    #     axes = axes.ravel()  # Flatten the axes for easier indexing
    #
    #     # Iterate through the flags and generate corresponding plots
    #     plot_index = 0
    #     for plot_name, plot_flag in self.plots.items():
    #         if plot_flag:
    #             if plot_name in ['generate_wordcloud', 'generate_logos']:
    #                 axes[plot_index].axis('off')  # Turn off axes for word clouds and logos
    #             else:
    #                 axes[plot_index].axis('on')
    #                 self._generate_individual_plot(plot_name, axes[plot_index])
    #             plot_index += 1
    #
    #     # Remove any remaining empty subplots
    #     for i in range(plot_index, len(axes)):
    #         fig.delaxes(axes[i])
    #
    #     # Adjust subplot layout
    #     plt.tight_layout()
    #
    #     # Show the combined visualization
    #     plt.show()

    # def generate_visualizations(self):
    #     """
    #     Generate visualizations for the MSA data and display them in subplots.
    #     """
    #     # Define the order of plotting methods
    #     plot_order = [
    #         'cleanse_heatmaps',
    #         'label_sequences',
    #         'generate_wordclouds',
    #         'select_features',
    #         'select_residues',
    #         'generate_logos',
    #     ]
    #
    #     # Define the mapping of plotting methods to their corresponding functions
    #     plot_method_mapping = {
    #         'cleanse_data': self._plot_cleanse_heatmaps,
    #         'label_sequences': self._plot_sequence_labels,
    #         'generate_wordclouds': self._plot_wordclouds,
    #         'select_features': self._plot_feature_selection,
    #         'select_residues': self._plot_selected_residues,
    #         'generate_logos': self._plot_logos,
    #     }
    #
    #     # Create subplots for visualizations
    #     num_plots = len(plot_order)
    #     num_cols = 2  # Number of columns in the subplots grid
    #     num_rows = (num_plots + num_cols - 1) // num_cols
    #     fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), constrained_layout=True)
    #
    #     # Flatten the 2D axes array if there's only one row
    #     if num_rows == 1:
    #         axes = axes.flatten()
    #
    #     # Generate individual plots
    #     plot_index = 0
    #     for plot_name in plot_order:
    #         if plot_name in plot_method_mapping:
    #             plot_method = plot_method_mapping[plot_name]
    #             if plot_index < len(axes):
    #                 ax = axes[plot_index]
    #                 plot_method(ax)
    #             plot_index += 1
    #
    #     # Remove any remaining empty subplots
    #     for i in range(plot_index, len(axes)):
    #         fig.delaxes(axes[i])
    #
    #     # Adjust subplot layout
    #     plt.tight_layout()
    #
    #     # Display the plots
    #     plt.show()

    # @staticmethod
    # def henikoff(data):
    #     """
    #     Calculate sequence weights using the Henikoff & Henikoff algorithm.
    #
    #     Parameters:
    #         data (pd.DataFrame): The MSA data.
    #
    #     Returns:
    #         pd.Series: A Series containing the calculated weights.
    #     """
    #     data_array = data.to_numpy()  # Convert DataFrame to NumPy array
    #     size, length = data_array.shape
    #     weights = []
    #     for seq_index in range(size):
    #         row = data_array[seq_index, :]
    #         unique_vals, counts = np.unique(row, return_counts=True)
    #         k = len(unique_vals)
    #         matrix_row = 1. / (k * counts)
    #         weights.append(np.sum(matrix_row) / length)
    #     return pd.Series(weights, index=data.index)
    #
    # def get_clusters(self, selected_columns=None, threshold=0.9, plot=False, export_tree=False, prefix='output'):
    #     """
    #     Cluster sequences in the MSA data.
    #
    #     Parameters:
    #         selected_columns (list, optional): list of previously selected MSA columns to base on.
    #         threshold (float, optional): The threshold for selecting important residues (default is 0.9).
    #         plot (bool, optional): Whether to plot the clustering results (default is False).
    #         export_tree (bool, optional): Whether to export the clustering dendrogram (default is False).
    #         prefix (str, optional): The prefix for the output dendrogram file (default is 'output').
    #     """
    #     if not selected_columns:
    #         # Perform feature selection on data
    #         self.select_features()
    #         # Calculate cumulative sum of importance
    #         cumulative_importance = np.cumsum(self.sorted_importance) / np.sum(self.sorted_importance)
    #         # Find the index where cumulative importance exceeds or equals 0.9
    #         index = np.where(cumulative_importance >= threshold)[0][0]
    #         # Get the values from sorted_features up to the index
    #         selected_columns = self.sorted_importance.index[:index + 1].values
    #         # Filter the selected features to get the most important residues
    #     selected_residues = [x for x in self.selected_features if int(x.split('_')[0]) in selected_columns]
    #     df_res = self.analysis.column_coordinates(self.data[selected_columns]).loc[selected_residues]
    #     # Get sequence weights through Henikoff & Henikoff algorithm
    #     weights = self.henikoff(self.data[selected_columns])
    #     # Create an empty graph
    #     g = nx.Graph()
    #     for idx in df_res.index:
    #         col, aa = idx.split('_')
    #         col = int(col)
    #         rows = self.data[selected_columns].index[self.data[selected_columns][col] == aa].tolist()
    #         # Filter and sum values based on valid indices
    #         p = weights.iloc[[i for i in rows if i < len(weights)]].sum()
    #         # Add a node with attributes
    #         g.add_node(
    #             f'{seq3(aa)}{col}',
    #             idx=idx,
    #             aa=aa,
    #             col=col,
    #             coord=(
    #                 df_res.loc[idx, 0],
    #                 df_res.loc[idx, 1]
    #             ),
    #             rows=rows,
    #             p=p
    #         )
    #     nodelist = sorted(g.nodes(), key=lambda x: g.nodes[x]['p'], reverse=True)
    #     df_res = df_res.loc[[g.nodes[u]['idx'] for u in nodelist]]
    #     df_res.columns = ['x_mca', 'y_mca']
    #     df_res = df_res.copy()
    #     # Generate pairwise combinations
    #     pairwise_comparisons = list(combinations(g.nodes, 2))
    #     # Add edges to the graph based on pairwise calculation of Jaccard's dissimilarity (1 - similarity)
    #     for u, v in pairwise_comparisons:
    #         asymmetric_distance = set(g.nodes[u]['rows']) ^ set(g.nodes[v]['rows'])
    #         union = set(g.nodes[u]['rows']) | set(g.nodes[v]['rows'])
    #         weight = float(
    #             weights.iloc[[i for i in list(asymmetric_distance) if i < len(weights)]].sum()
    #         ) / float(
    #             weights.iloc[[i for i in list(union) if i < len(weights)]].sum()
    #         ) if g.nodes[u]['col'] != g.nodes[v]['col'] else 1.
    #         g.add_edge(
    #             u,
    #             v,
    #             weight=weight
    #         )
    #     # Generate a distance matrix
    #     d = nx.to_numpy_array(g, nodelist=nodelist)
    #     # Apply OPTICS on the points
    #     optics = OPTICS(metric='precomputed', min_samples=3)
    #     optics.fit(d)
    #     # Retrieve cluster labels
    #     cluster_labels = optics.labels_
    #     unique_labels = np.unique(cluster_labels)
    #     clusters = [[] for _ in unique_labels]
    #     for i, cluster_label in enumerate(unique_labels):
    #         cluster_indices = np.where(cluster_labels == cluster_label)[0]
    #         cluster_nodes = [nodelist[idx] for idx in cluster_indices]
    #         clusters[i] = cluster_nodes
    #     adhesion = []
    #     for row_idx in self.data.index:
    #         temp = []
    #         for cluster in clusters:
    #             count = 0
    #             for node in cluster:
    #                 if self.data.loc[row_idx, g.nodes[node]['col']] == g.nodes[node]['aa']:
    #                     count += 1
    #             temp.append(float(count) / float(len(cluster)))
    #         adhesion.append(np.array(temp))
    #     adhesion = np.array(adhesion)
    #     df_adh = pd.DataFrame(adhesion)
    #     df_adh.columns = [f"Cluster {label + 1}" if label >= 0 else "Noise" for label in unique_labels]
    #     itemgetter_func = itemgetter(*self.data.index)
    #     seq_ids = itemgetter_func(self.raw_data.index)
    #     df_adh.index = seq_ids
    #     # Run clustermap with potential performance improvement
    #     z = fastcluster.linkage(df_adh, method='ward')
    #     if export_tree:
    #         write_dendrogram(z, prefix)
    #     # List to store silhouette scores
    #     silhouette_scores = []
    #     # Define a range of possible values for k
    #     k_values = range(2, 10)
    #     # Calculate silhouette score for each value of k
    #     for k in k_values:
    #         labels = fcluster(z, k, criterion='maxclust')
    #         silhouette_scores.append(silhouette_score(df_adh, labels))
    #     # Find the index of the maximum silhouette score
    #     best_index = np.argmax(silhouette_scores)
    #     # Get the best value of k
    #     best_k = k_values[best_index]
    #     # Get the cluster labels for each sequence in the MSA
    #     sequence_labels = fcluster(z, best_k, criterion='maxclust')
    #     if plot:
    #         # fig = plt.figure(figsize=(12, 7))  # Create the main figure container
    #         # Plot the distance matrix
    #         fig, ax = plt.subplots()
    #         im = ax.imshow(d, cmap='viridis')
    #         # Add a colorbar
    #         ax.figure.colorbar(im, ax=ax)
    #         self.plots['get_clusters'].append(plt)
    #         ordering = optics.ordering_
    #         # Perform MDS to obtain the actual points
    #         mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress='auto')
    #         points = mds.fit_transform(d)
    #         df_res['x_mds'] = points[:, 0]
    #         df_res['y_mds'] = points[:, 1]
    #         df_res['label'] = cluster_labels
    #         df_res = df_res.copy()
    #         # Plot the ordering analysis
    #         plt.figure(figsize=(12, 4))
    #         plt.bar(range(len(ordering)), ordering, width=1., color='black')
    #         plt.xlim([0, len(ordering)])
    #         plt.xlabel('Points')
    #         plt.ylabel('Reachability Distance')
    #         plt.title('Ordering Analysis')
    #         # Show the plots
    #         self.plots['get_clusters'].append(plt)
    #         # Create a figure with two subplots
    #         fig, axs = plt.subplots(1, 2, figsize=(12, 7))
    #         # Plot for MCA
    #         axs[0].set_xlabel('MCA Dimension 1', fontsize=14)
    #         axs[0].set_ylabel('MCA Dimension 2', fontsize=14)
    #         axs[0].set_title('Residues from MCA Colored by Cluster', fontsize=16)
    #         # Plot for MDS
    #         axs[1].set_xlabel('MDS Dimension 1', fontsize=14)
    #         axs[1].set_ylabel('MDS Dimension 2', fontsize=14)
    #         axs[1].set_title('Residues from MDS Colored by Cluster', fontsize=16)
    #         # Plot on the subplots
    #         x_mca = df_res['x_mca']
    #         y_mca = df_res['y_mca']
    #         x_mds = df_res['x_mds']
    #         y_mds = df_res['y_mds']
    #         labels = df_res['label']
    #         unique_labels = np.unique(labels)
    #         color_map = cm.get_cmap('tab10', len(unique_labels))
    #         handles = []
    #         all_labels = []
    #         for i, cluster_label in enumerate(unique_labels):
    #             if cluster_label == -1:
    #                 color = 'grey'
    #                 marker = 'x'
    #                 label = 'Noise'
    #                 alpha = 0.25
    #             else:
    #                 color = color_map(i)
    #                 marker = 'o'
    #                 label = f'Cluster {cluster_label + 1}'
    #                 alpha = None
    #             scatter = axs[0].scatter(
    #                 x_mca[labels == cluster_label],
    #                 y_mca[labels == cluster_label],
    #                 color=color,
    #                 marker=marker,
    #                 label=label,
    #                 alpha=alpha
    #             )
    #             axs[1].scatter(
    #                 x_mds[labels == cluster_label],
    #                 y_mds[labels == cluster_label],
    #                 color=color,
    #                 marker=marker,
    #                 label=label,
    #                 alpha=alpha
    #             )
    #             # Collect the scatter plot handles and labels
    #             handles.append(scatter)
    #             all_labels.append(label)
    #         # Rearrange labels to fill one row first and then the second row
    #         n_cols = 6  # Number of columns in the legend
    #         all_labels_reordered = [all_labels[i::n_cols] for i in range(n_cols)]
    #         all_labels_reordered = sum(all_labels_reordered, [])  # Flatten the nested list
    #         # Create a single legend for both subplots with reordered labels
    #         fig.legend(handles, all_labels_reordered, bbox_to_anchor=(0.5, 0.0), loc='upper center', borderaxespad=0,
    #                    ncol=n_cols, fontsize=12)
    #         # Adjust the layout to accommodate the legend
    #         fig.tight_layout(rect=[0, 0.05, 1, 0.9])
    #         cbar_kws = {"orientation": "horizontal"}
    #         g = sns.clustermap(df_adh.drop('Noise', axis=1), col_cluster=False, yticklabels=False,
    #                            cbar_pos=(.4, .9, .4, .02), cbar_kws=cbar_kws, figsize=(5, 5))
    #         # Show the plot
    #         self.plots['get_clusters'].append(plt)
    #     self.clusters = {label: cluster for label, cluster in zip(unique_labels, clusters)}
    #     self.labels = sequence_labels

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

# import matplotlib.cm as cm
# from sklearn.cluster import OPTICS
# import networkx as nx
# from itertools import combinations
# from sklearn.manifold import MDS
# from scipy.cluster.hierarchy import to_tree
# from operator import itemgetter
# from functools import lru_cache
# import random

# # Define a dictionary for amino acid stereochemistry
# STHEREOCHEMISTRY = {
#     'Aliphatic': ['G', 'A', 'V', 'L', 'I'],
#     'Amide': ['N', 'Q'],
#     'Aromatic': ['F', 'Y', 'W'],
#     'Basic': ['H', 'K', 'R'],
#     'Big': ['M', 'I', 'L', 'K', 'R'],
#     'Hydrophilic': ['R', 'K', 'N', 'Q', 'P', 'D'],
#     'Median': ['E', 'V', 'Q', 'H'],
#     'Negatively charged': ['D', 'E'],
#     'Non-polar': ['F', 'G', 'V', 'L', 'A', 'I', 'P', 'M', 'W'],
#     'Polar': ['Y', 'S', 'N', 'T', 'Q', 'C'],
#     'Positively charged': ['K', 'R'],
#     'Similar (Asn or Asp)': ['N', 'D'],
#     'Similar (Gln or Glu)': ['Q', 'E'],
#     'Small': ['C', 'D', 'P', 'N', 'T'],
#     'Tiny': ['G', 'A', 'S'],
#     'Very hydrophobic': ['L', 'I', 'F', 'W', 'V', 'M'],
#     'With hydroxyl': ['S', 'T', 'Y'],
#     'With sulfur': ['C', 'M']
# }


# def get_newick(node, parent_dist, leaf_names, newick=""):
#     """
#     Recursively construct a Newick string representation of a hierarchical tree.
#
#     Parameters:
#         node (TreeNode): The current node in the hierarchical tree.
#         parent_dist (float): The distance between the current node and its parent node.
#         leaf_names (list): A list of leaf names corresponding to the tree nodes.
#         newick (str, default=""): The Newick string representation of the hierarchical tree.
#
#     Returns:
#         A Newick string representation of the hierarchical tree.
#     """
#     if node.is_leaf():
#         return f"{leaf_names[node.id]}:{parent_dist - node.dist:.2f}{newick}"
#     else:
#         if len(newick) > 0:
#             newick = f"):{parent_dist - node.dist:.2f}{newick}"
#         else:
#             newick = ");"
#         newick = get_newick(node.get_left(), node.dist, leaf_names, newick)
#         newick = get_newick(node.get_right(), node.dist, leaf_names, f",{newick}")
#         newick = f"({newick}"
#         return newick
#
#
# def write_dendrogram(linkage_object, headers, prefix='output'):
#     """
#     Write the Newick representation of a dendrogram to a file.
#
#     Parameters:
#         linkage_object (object): The linkage object resulting from hierarchical clustering.
#         headers (list): A list of headers corresponding to the data points being clustered.
#         prefix (str, default='output'): The prefix for the output dendrogram file.
#
#     This function writes the Newick representation of a dendrogram to a file using the provided
#     linkage object and headers.
#     """
#     tree = to_tree(linkage_object, False)
#     with open(f'{prefix}_dendrogram.nwk', 'w') as outfile:
#         outfile.write(get_newick(tree, "", tree.dist, headers))
#
#
# def cached_property(method):
#     """
#     Create a cached property using the lru_cache decorator.
#
#     Parameters:
#         method (callable): A method that calculates the property value.
#
#     Returns:
#         The cached property value.
#     """
#
#     @property
#     @lru_cache()
#     def wrapper(self):
#         return method(self)
#
#     return wrapper


# %%
