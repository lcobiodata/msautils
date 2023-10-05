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
import warnings
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
import os
import sys
import argparse
import pickle

def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class MSA(pd.DataFrame):
    """
    A class for processing and analyzing Multiple Sequence Alignments (MSA).
    """
    def __init__(self, msa_file=None, msa_format="fasta", metadata=None, *args, **kwargs):
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
        self.metadata = metadata
        self.positions_map = None  # Mapping of positions in the MSA
        self.dirty = None  # Dirty data (before cleansing)
        self.clean = None  # Clean data (after cleansing)
        self.unique = None  # Processed MSA data
        self.mca = None  # Analysis results
        self.coordinates = None  # Reduced data coordinates
        self.labels = None  # Sequence labels or clusters
        self.encoded = None # One-hot encoded data set
        self.sorted_importance = None  # Sorted feature importances
        self.selected_features = None  # Selected features
        self.selected_columns = None  # Selected columns from the data
        self.profiles = None  # Residue profiles
        self.wordcloud_data = None  # Word cloud data
        self.logos_data = None  # Logo data

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
            raise ValueError("The MSA data frame is empty. Ensure you've loaded the data correctly.")
        
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
        print(f"self.dirty.shape after replacing specified characters with NaN: {self.dirty.shape}")

        # Create a copy of 'dirty' data as the 'clean' data
        self.clean = self.dirty.copy()
        print(f"self.clean.shape right after creating a copy of 'dirty' data as the 'clean' data: {self.clean.shape}")

        # Calculate the minimum number of non-NaN values for rows
        min_rows = int(threshold * self.clean.shape[0])
        print(f"min_rows: {min_rows}")

        # Remove columns with NaN values above the threshold
        self.clean.dropna(thresh=min_rows, axis=1, inplace=True)
        print(f"self.clean.shape after removing columns with NaN values above the threshold: {self.clean.shape}")

        # Calculate the minimum number of non-NaN values for columns
        min_cols = int(threshold * self.clean.shape[1])
        # print(f"min_cols: {min_cols}")

        # Remove rows with NaN values above the threshold
        self.clean.dropna(thresh=min_cols, axis=0, inplace=True)
        print(f"self.clean.shape after removing rows with NaN values above the threshold: {self.clean.shape}")

        # Reset the index, drop duplicates, and fill NaN values with '-'
        self.unique = self.clean.reset_index(drop=True) \
            .drop_duplicates() \
            .fillna('-') \
            .copy()

        # If plotting is enabled, plot heatmaps
        if plot:
            self._plot_cleanse_heatmaps(save=save, show=show)

    def _plot_cleanse_heatmaps(self, save=False, show=False):
        """
        Generate and display cleansing heatmaps plot on the specified axes.
        """
        # Create a figure with two subplots (heatmaps)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Create the heatmap before cleansing on the first Axes
        ax1 = axes[0]  # First subplot
        heatmap_before = ax1.imshow(self.dirty.isna().astype(int), cmap='viridis', aspect='auto', extent=[0, 1, 0, 1])

        # Create the heatmap after cleansing on the second Axes
        ax2 = axes[1]  # Second subplot
        heatmap_after = ax2.imshow(self.clean.isna().astype(int), cmap='viridis', aspect='auto', extent=[0, 1, 0, 1])

        # Create a shared color bar axis
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust the position and size as needed

        # Add color bars to the right of the heatmaps with the shared color bar axis
        cbar_before = plt.colorbar(heatmap_before, cax=cax)
        cbar_before.set_label('Gaps (Indels)')
        cbar_after = plt.colorbar(heatmap_after, cax=cax)  # Use the same color bar axis
        cbar_after.set_label('Gaps (Indels)')

        ax1.axis('off')  # Turn off axis labels for the first subplot
        ax2.axis('off')  # Turn off axis labels for the second subplot

        if show:
            plt.show()

        if save:
            plt.savefig("./output/cleanse_heatmaps.png", dpi=300)

    def reduce(self, plot=False, save=False, show=True, *args, **kwargs):
        """
        Perform Multidimensional Correspondence Analysis (MCA) on the MSA data to reduce dimensionality.

        Parameters:
            plot (bool, optional): If True, the MCA results will be plotted (default is False).
            save (bool, optional): If True and `plot` is also True, the plotted results will be saved to a file.
            show (bool, optional): If True and `plot` is also True, the plotted results will be displayed.
            *args, **kwargs: Additional arguments and keyword arguments passed to the MCA.

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
            # Option 1: Automatically call cleanse
            self.cleanse()
            
            # Option 2: Raise an exception (uncomment the below line and comment out Option 1 if you want this)
            # raise ValueError("self.unique is not set. Make sure to run self.cleanse() before calling self.reduce()")

        # Perform MCA
        self.mca = MCA(*args, **kwargs)
        self.mca.fit(self.unique)

        self.coordinates = np.array(self.mca.transform(self.unique))

        if plot:
            self._plot_mca(save=save, show=show)

    def _plot_mca(self, save=False, show=True):
        """
        Private method to plot the results of Multidimensional Correspondence Analysis (MCA).

        Based on the MCA results, this method plots a scatter plot with sequences and residues overlaid.
        When the number of rows exceeds a predefined limit (ROW_LIMIT), the method performs proportional sampling 
        based on cluster sizes. This is done to ensure clarity in the visualization and optimize the performance.
        
        Parameters:
        - save (bool): If True, the plot will be saved to a specified path.
        - show (bool): If True, the plot will be displayed immediately.

        Notes:
        - If the number of unique sequences exceeds the ROW_LIMIT, the method resorts to proportional sampling.
        - For the sampling, if clustering labels are not previously generated, the method will execute the clustering 
          step and then sample according to the cluster sizes.
        
        Usage:
        msa = MSA('example.fasta')
        msa.map_positions()
        msa.cleanse()
        msa.reduce()
        msa._plot_mca(save=True, show=False) # This will only save the plot and not display it.
        """
        ROW_LIMIT = 5000

        # Define the dummy function do_not_show
        def do_not_show(*args, **kwargs):
            pass
        
        # Check if rows exceed the limit
        if len(self.unique) > ROW_LIMIT:
            params = {'plot': False, 'save': False, 'show': False}
            sample = self.unique.sample(n=5000, random_state=42)
            obj = MSA(data=sample, index=sample.index)
            obj.cleanse(**params)
            obj.reduce(**params)
        else:
            obj = self
        
        # If we don't want to show the plot, redirect plt.show()
        if not show:
            original_show = plt.show
            plt.show = do_not_show

        # Plot the data
        try:
            obj.mca.plot(
                obj.unique,
                x_component=0,
                y_component=1
            )
            # If you want to save the plot
            if save:
                plt.savefig("./output/mca_plot.png", dpi=300)
        except Exception as e:
            warnings.warn(f"Unable to plot mca results: {e}", UserWarning)
        finally:
            pass

        # If we saved the show, revert plt.show() back to its original state
        if not show:
            plt.show = original_show

    def cluster(self, min_clusters=2, max_clusters=10, method='single-linkage', plot=False, save=False, show=False, **kwargs):
        """
        Cluster the MSA data and obtain cluster labels.

        Parameters:
            min_clusters (int, optional): Minimum number of clusters (default is 2).
            max_clusters (int, optional): Maximum number of clusters (default is 10).
            method (str, optional): Clustering method ('k-means' or 'single-linkage') (default is 'single-linkage').
            plot (bool, optional): If True, the clustering results will be plotted (default is False).
            save (bool, optional): If True and `plot` is also True, the plotted results will be saved to a file.
            show (bool, optional): If True and `plot` is also True, the plotted results will be displayed.
            **kwargs: Additional keyword arguments passed to the clustering method.

        Notes:
        - This method performs clustering on the MSA data and assigns cluster labels to sequences.
        - Clustering can be done using either k-means or single-linkage methods.
        - The optimal number of clusters is determined using silhouette scores.
        - The cluster labels are stored in the 'labels' attribute of the MSA object.

        Example:
        msa = MSA('example.fasta')
        msa.map_positions()
        msa.cleanse()
        msa.cluster(method='single-linkage', min_clusters=3, plot=True, save=True)
        """
        # Check attribute dependencies and run the dependent method if needed
        if not hasattr(self, 'coordinates') or  self.coordinates is None:
            self.reduce()  # Assuming this is the method that populates self.coordinates

        if method not in ['k-means', 'single-linkage']:
            raise ValueError("method must be 'k-means' or 'single-linkage")

        # Define a range of potential numbers of clusters to evaluate
        k_values = range(min_clusters, max_clusters)
        # Perform clustering for different numbers of clusters and compute silhouette scores
        model, silhouette_scores = None, []
        for k in range(min_clusters, max_clusters + 1):
            if method == 'k-means':
                model = KMeans(n_clusters=k, n_init=max_clusters)  # Set n_init explicitly
                model.fit(self.coordinates)
                labels = model.labels_
                score = silhouette_score(self.coordinates, labels)
                silhouette_scores.append(score)
            elif method == 'single-linkage':
                # Run clustermap with potential performance improvement
                model = fastcluster.linkage(self.coordinates, method='ward')
                # Calculate silhouette score for each value of k
                labels = fcluster(model, k, criterion='maxclust')
                silhouette_scores.append(silhouette_score(self.coordinates, labels))

        # Find the index of the maximum silhouette score
        best_index = np.argmax(silhouette_scores)
        # Perform the actual modeling depending on the chosen algorithm
        if method == 'k-means':
            # Find the best number of clusters based on the highest silhouette score
            best_num_clusters = best_index + min_clusters
            # Perform clustering with the best number of clusters
            model = KMeans(n_clusters=best_num_clusters, n_init=max_clusters)
            model.fit(self.coordinates)
            self.labels = model.labels_
        elif method == 'single-linkage':
            # Get the best value of k
            best_k = k_values[best_index]
            # Get the cluster labels for each sequence in the MSA
            self.labels = fcluster(model, best_k, criterion='maxclust')

        if plot:
            self._plot_wordclouds(save=save, show=show, **kwargs)

    def _plot_wordclouds(self, metadata=None, column='Protein names', save=False, show=False):
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
        # If metadata is None, default to self.metadata
        metadata = metadata or self.metadata

        if metadata is not None:
            
            # Read the TSV file into a DataFrame
            try:
                metadata = pd.read_csv(metadata, delimiter='\t')
            except FileNotFoundError:
                raise

            self.wordcloud_data = {}

            # Process data for each cluster
            for label in np.unique(self.labels):
                indices = self.unique.iloc[self.labels == label].index
                headers = self.index[indices]
                entry_names = [header.split('/')[0] for header in headers]

                result = metadata[metadata['Entry Name'].isin(entry_names)].copy()

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
            fig, axs = plt.subplots(num_clusters, 1, figsize=(10, 6 * num_clusters))

            for i, (label, wordcloud_text) in enumerate(self.wordcloud_data.items()):
                ax = axs[i] if num_clusters > 1 else axs

                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white'
                ).generate(wordcloud_text)

                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')

            if show:
                plt.show()

            if save:
                plt.savefig("./output/wordclould.png")

        else:
            warnings.warn("metadata is not provided and self.metadata is not set. Please provide the 'metadata' argument or set self.metadata.", UserWarning)

    def select_features(self, n_estimators=None, random_state=None, plot=False, save=False, show=False):
        """
        Select important features (residues) from the MSA data.

        Parameters:
            n_estimators (int, optional): Parameter n_estimators for RandomForest.
            random_state: (int, optional): Parameter random_state for RandomForest.
            plot (bool, optional): Whether to plot feature selection results (default is False).
            save (bool, optional): Whether to save feature selection results (default is False).
            show (bool, optional): Whether to show feature selection results (default is False).
        """
        # Check attribute dependencies and run the dependent method if needed
        if not hasattr(self, 'labels') or self.labels is None:
            self.cluster()  # Assuming this is the method that populates self.labels

        # Extract X (features) and y (labels)
        x = self.unique
        y = pd.get_dummies(self.labels).astype(int) if len(np.unique(self.labels)) > 2 else self.labels

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

        # Store the encoded data frame
        self.encoded = x_encoded_df

        # Store the selected features and their importances
        self.selected_features = selected_features
        self.sorted_importance = sorted_importance

        if plot:
            self._plot_clustermap(save=save, show=show)
            self._plot_pareto(save=save, show=show)

    def _plot_pareto(self, save=False, show=True):
        """
        Generate and display a Pareto plot of the feature importances.

        Parameters:
            save (bool, optional): If True, the plotted results will be saved to a file (default is False).
            show (bool, optional): If True, the plotted results will be displayed (default is True).
        
        Notes:
        - This method produces a Pareto plot that visualizes the distribution of feature importances.
        - This plot can be useful in understanding which features contribute the most to model performance.
        """
        sorted_features = self.sorted_importance.index

        # Create a figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(16, 4))

        # Bar chart of percentage importance
        xvalues = range(len(sorted_features))
        ax1.bar(xvalues, self.sorted_importance, color='cyan')
        ax1.set_ylabel('Summed Importance', fontsize=16)
        ax1.tick_params(axis='y', labelsize=12)

        # Create a second y-axis for the line chart
        ax2 = ax1.twinx()

        # Line chart of cumulative percentage importance
        ax2.plot(xvalues, np.cumsum(self.sorted_importance) / np.sum(self.sorted_importance), color='magenta',
                 marker='.')
        ax2.set_ylabel('Cumulative Importance', fontsize=16)
        ax2.tick_params(axis='y', labelsize=12)

        # Rotate x-axis labels for better visibility
        plt.xticks(xvalues, sorted_features)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)

        if show:
            plt.show()

        if save:
            plt.savefig("./output/pareto_chart.png")

    def _plot_clustermap(self, save=False, show=False):
        """
        Generate and display a clustermap of the encoded data using seaborn.        
        """
        plt.figure(figsize=(16, 16))
        g = sns.clustermap(self.encoded, method="single", cmap="viridis", standard_scale=1)

        # Hide xticks and yticks
        g.ax_heatmap.set_xticks([])
        g.ax_heatmap.set_yticks([])

        if save:
            plt.savefig("./output/clustermap.png")

        if show:
            plt.show()

    def select_residues(self, threshold=0.9, top_n=None, plot=False, save=False, show=True, **kwargs):
        """
        Select and store residues that are candidates for Specificity-determining Positions (SDPs) from the MSA data.

        Parameters:
            threshold (float, optional): The threshold for selecting residues based on their importance. Default is 0.9.
            top_n (int, optional): Selects the top N residues based on their importance. If set to None, selection is based on the threshold. Default is None.
            plot (bool, optional): If True, visual representation of the selected residues will be plotted. Default is False.
            save (bool, optional): If True, the plot will be saved to a file. This parameter is considered only if `plot` is True. Default is False.
            show (bool, optional): If True, the plot will be displayed. This parameter is considered only if `plot` is True. Default is True.

        Notes:
        - This method selects residues that are candidates for Specificity-determining Positions (SDPs) based on their importance.
        - The importance can be determined through various methods (e.g., feature importance from a classifier, conservation score).
        - Selected residues can offer insights into the functional or structural significance in the protein family.
        """
        # Check attribute dependencies and run the dependent method if needed
        if not hasattr(self, 'selected_features') or self.selected_features is None or not hasattr(self, 'sorted_importance') or self.sorted_importance is None:

            self.select_features(plot=False, save=False, show=False, **kwargs)  # Assuming this is the method that populates self.sorted_importance

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
            self._plot_perceptual_map(save=save, show=show)
            self._plot_logos(save=save, show=show, **kwargs)

    def _plot_perceptual_map(self, save=False, show=True):
        """
        Generate and display a perceptual map of selected residues.

        Parameters:
            save (bool, optional): If True, the generated perceptual map will be saved to a file. Default is False.
            show (bool, optional): If True, the generated perceptual map will be displayed immediately. Default is True.

        Notes:
        - This method is designed to visualize the selected residues on a perceptual map. 
        - The perceptual map can offer insights into the distribution, clustering, or relationships of the residues based on certain metrics or dimensions.
        - This is particularly useful for understanding the spatial arrangement or similarity of residues in some context.
        """
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

        # Create figure
        plt.figure(figsize=(8, 6))

        # Create legends for cluster labels
        unique_labels = np.unique(self.labels)
        legend_handles = []
        # Iterate through unique labels
        for label in unique_labels:
            # Find indices where the labels array matches the current label
            indices = np.where(self.labels == label)[0]
            plt.scatter(
                self.coordinates[indices, 0],
                self.coordinates[indices, 1],
                color=plt.cm.viridis(label / len(unique_labels)),
                alpha=0.5
            )
            legend_handles.append(
                plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}', markersize=10,
                           markerfacecolor=plt.cm.viridis(label / len(unique_labels)))
            )

        # Scatter plot of labeled residues
        plt.scatter(df_res[0], df_res[1], marker='*', color='black', s=50)

        # Annotate labeled residues
        for i, (x, y) in enumerate(zip(df_res[0], df_res[1])):
            plt.annotate(df_res.index[i], (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

        legend_handles.append(
            plt.Line2D([0], [0], marker='*', color='k', label='Selected Residues', markersize=10)
        )

        # Set labels and title
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

        plt.legend(handles=legend_handles, title='Sequence Clusters')

        plt.grid()

        if show:
            plt.show()

        if save:
            plt.savefig("./output/perceptual_map.png")

    def _plot_logos(self, color_scheme='NajafabadiEtAl2017', plot=False, save=False, show=False, **kwargs):
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
        if not plot:
            return

        self.logos_data = {}  # Initialize logos_data as an empty dictionary
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            sub_msa = self.unique[sorted(self.selected_columns)].iloc[self.labels == label]

            # Calculate sequence frequencies for each position
            data = sub_msa.T.apply(lambda col: col.value_counts(normalize=True), axis=1).fillna(0)

            # Store the seq_logo in logos_data with the label as the key
            self.logos_data[label] = data

        color_schemes = lm.list_color_schemes()
        color_schemes_list = sorted(color_schemes.loc[color_schemes.characters == 'ACDEFGHIKLMNPQRSTVWY'].color_scheme.values)

        if color_scheme not in color_schemes_list:
            raise ValueError(f"color scheme must be in {color_schemes_list}")

        n_labels = len(unique_labels)
        fig, axs = plt.subplots(nrows=int(np.ceil(n_labels / 2)), ncols=2, figsize=(10, 5 * n_labels))
        
        # Ensure axs is always a 2D array, even when n_labels = 1
        if n_labels == 1:
            axs = np.array([[axs]])

        for i, (label, data) in enumerate(self.logos_data.items()):
            msa_columns = data.index.tolist()
            data = data.reset_index(drop=True)
            
            ax = axs[i // 2, i % 2]

            # Create a sequence logo from the DataFrame, passing in the current axis
            seq_logo = lm.Logo(data, ax=ax, color_scheme=color_scheme, vpad=.1, width=.8)

            # Customize the appearance of the logo
            seq_logo.style_spines(visible=False)
            
            ax.set_xticks(range(len(msa_columns)))
            ax.set_xticklabels(msa_columns, fontsize=12)
            ax.set_title(f"Label: {label}")

        # Remove any unused subplots
        if n_labels % 2 != 0:
            axs[-1, -1].axis('off')

        plt.tight_layout()

        if save:
            plt.savefig(f"./output/sdp_combined_logo.png")

        if show:
            plt.show()


def _should_plot(method, args):
    """Utility function to check if a particular method should plot."""
    return method in args.plot_methods

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

        # Initializes MSA object
        msa = MSA(args.data, metadata=args.metadata)
        
        # Downloading nltk resources is only necessary for _plot_weblog, which requires metadata
        if args.metadata:
            download_nltk_resources()
        
        msa.map_positions()

        msa.cleanse(plot=True, save=True, show=(not args.hide))
        msa.reduce(plot=True, save=True, show=(not args.hide))
        msa.cluster(method='single-linkage', min_clusters=3, plot=True, save=True, show=(not args.hide))
        msa.select_features(n_estimators=1000, random_state=42, plot=True, save=True, show=(not args.hide))
        msa.select_residues(top_n=3, plot=True, save=True, show=(not args.hide))

        if args.export:
            filename = os.path.basename(args.data)
            basename, _ = os.path.splitext(filename)
            with open(f"./output/{basename}.pkl", "wb") as f:
                pickle.dump(msa, f)

        return True

    except Exception as e:
        raise

    print("An unexpected error occurred.")
    return False

if __name__ == "__main__":
    try:
        sys.exit(0 if main() else 1)
    except Exception as e:
        import traceback
        traceback.print_exc()  # This will print the full traceback
        print(f"An error occurred: {type(e).__name__} - {e}")
        sys.exit(2)

