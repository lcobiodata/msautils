# Table of Contents

* [protlearn](#protlearn)
  * [MSA](#protlearn.MSA)
    * [\_\_init\_\_](#protlearn.MSA.__init__)
    * [parse\_msa\_file](#protlearn.MSA.parse_msa_file)
    * [map\_positions](#protlearn.MSA.map_positions)
    * [cleanse\_data](#protlearn.MSA.cleanse_data)
    * [reduce](#protlearn.MSA.reduce)
    * [label\_sequences](#protlearn.MSA.label_sequences)
    * [generate\_wordclouds](#protlearn.MSA.generate_wordclouds)
    * [select\_features](#protlearn.MSA.select_features)
    * [select\_residues](#protlearn.MSA.select_residues)
    * [generate\_logos](#protlearn.MSA.generate_logos)
  * [main](#protlearn.main)
  * [parse\_args](#protlearn.parse_args)

<a id="protlearn"></a>

# protlearn

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

<a id="protlearn.MSA"></a>

## MSA Objects

```python
class MSA(pd.DataFrame)
```

A class for processing and analyzing Multiple Sequence Alignments (MSA).

<a id="protlearn.MSA.__init__"></a>

#### \_\_init\_\_

```python
def __init__(*args, **kwargs)
```

Initialize the MSA object.

**Arguments**:

- `*args` - Variable-length positional arguments.
- `**kwargs` - Variable-length keyword arguments.

<a id="protlearn.MSA.parse_msa_file"></a>

#### parse\_msa\_file

```python
def parse_msa_file(msa_file, msa_format="fasta", *args, **kwargs)
```

Parse an MSA file and store the raw data in the MSA object.

**Arguments**:

- `msa_file` _str_ - The path to the MSA file.
- `msa_format` _str, optional_ - The format of the MSA file (default is "fasta").
- `*args` - Additional positional arguments to pass to SeqIO.parse.
- `**kwargs` - Additional keyword arguments to pass to SeqIO.parse.

<a id="protlearn.MSA.map_positions"></a>

#### map\_positions

```python
def map_positions()
```

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
  'Seq1/1-100': {1: 1, 2: 2, ...},
  'Seq2/101-200': {101: 1, 102: 2, ...},
...
}

This mapping is useful for downstream analysis that requires knowing the position
of residues in the MSA.

**Notes**:

  - Residues represented by '-' (indels/gaps) are not included in the mapping.
  

**Example**:

  msa = MSA()
  msa.parse_msa_file('example.fasta')
  msa.map_positions()
  
  Access the mapping:
  positions = msa.positions_map

<a id="protlearn.MSA.cleanse_data"></a>

#### cleanse\_data

```python
def cleanse_data(indel='-', remove_lowercase=True, threshold=.9, plot=False)
```

Cleanse the MSA data by removing columns and rows with missing values.

**Arguments**:

- `indel` _str, optional_ - The character representing gaps/indels (default is '-').
- `remove_lowercase` _bool, optional_ - Whether to remove lowercase characters (default is True).
- `threshold` _float, optional_ - The threshold for missing values (default is 0.9).
- `plot` _bool, optional_ - Whether to plot a heatmap of missing values (default is False).

<a id="protlearn.MSA.reduce"></a>

#### reduce

```python
def reduce(plot=False, *args, **kwargs)
```

Perform Multidimensional Correspondence Analysis (MCA) on the MSA data to reduce dimensionality.

**Arguments**:

- `plot` _bool, optional_ - Whether to plot the results (default is False).
  *args, **kwargs: Additional arguments and keyword arguments for the MCA.
  

**Notes**:

  - Multidimensional Correspondence Analysis (MCA) is used to reduce the dimensionality of the MSA data.
  - The MCA results are stored in the 'analysis' attribute of the MSA object.
  - The row coordinates after reduction are stored in the 'coordinates' attribute.
  

**Example**:

  msa = MSA()
  msa.parse_msa_file('example.fasta')
  msa.map_positions()
  msa.cleanse_data()
  msa.reduce(plot=True)

<a id="protlearn.MSA.label_sequences"></a>

#### label\_sequences

```python
def label_sequences(min_clusters=2,
                    max_clusters=10,
                    method='single-linkage',
                    plot=False)
```

Cluster the MSA data and obtain cluster labels.

**Arguments**:

- `min_clusters` _int, optional_ - Minimum number of clusters (default is 2).
- `max_clusters` _int, optional_ - Maximum number of clusters (default is 10).
- `method` _str, optional_ - Clustering method ('k-means' or 'single-linkage') (default is 'single-linkage').
- `plot` _bool, optional_ - Whether to plot the clustering results (default is False).
  

**Notes**:

  - This method performs clustering on the MSA data and assigns cluster labels to sequences.
  - Clustering can be done using either k-means or single-linkage methods.
  - The optimal number of clusters is determined using silhouette scores.
  - The cluster labels are stored in the 'labels' attribute of the MSA object.
  

**Example**:

  msa = MSA()
  msa.parse_msa_file('example.fasta')
  msa.map_positions()
  msa.cleanse_data()
  msa.label_sequences(method='single-linkage', min_clusters=3, plot=True)

<a id="protlearn.MSA.generate_wordclouds"></a>

#### generate\_wordclouds

```python
def generate_wordclouds(path_to_metadata=None,
                        column='Protein names',
                        plot=False)
```

Generate word cloud visualizations from protein names in a DataFrame.

**Arguments**:

- `path_to_metadata` _str, default=None_ - Path to metadata file in tsv format.
- `column` _str, default='Protein names'_ - The name of the column in the DataFrame containing protein names.
- `plot` _bool, default=False_ - Whether to plot word clouds.
  
  This method extracts substrate and enzyme names from the specified column using regular expressions,
  normalizes the names, and creates word cloud plots for each cluster of sequences.
  

**Example**:

  msa = MSA()
  msa.parse_msa_file('example.fasta')
  msa.map_positions()
  msa.cleanse_data()
  msa.label_sequences(method='single-linkage', min_clusters=3)
  msa.generate_wordclouds(path_to_metadata='metadata.tsv', plot=True)

<a id="protlearn.MSA.select_features"></a>

#### select\_features

```python
def select_features(n_estimators=None, random_state=None, plot=False)
```

Select important features (residues) from the MSA data.

**Arguments**:

- `n_estimators` _int, optional_ - Parameter n_estimators for RandomForest.
- `random_state` - (int, optional): Parameter random_state for RandomForest.
- `plot` _bool, optional_ - Whether to plot feature selection results (default is False).

<a id="protlearn.MSA.select_residues"></a>

#### select\_residues

```python
def select_residues(threshold=0.9, top_n=None, plot=False)
```

Select and store residues to be candidates for Specificity-determining Positions (SDPs) from the MSA data.

**Arguments**:

- `threshold` _float, optional_ - The threshold for selecting residues based on importance (default is 0.9).
- `top_n` _int, optional_ - The top N residues to select based on importance (default is None).
- `plot` _bool, optional_ - Whether to plot the selected residues (default is False).

<a id="protlearn.MSA.generate_logos"></a>

#### generate\_logos

```python
def generate_logos(plot=False)
```

Generate logos from the MSA data for each cluster label.

**Arguments**:

- `plot` _bool, optional_ - Whether to plot the generated logos (default is False).

<a id="protlearn.main"></a>

#### main

```python
def main(args)
```

Main function to process the Multiple Sequence Alignment (MSA) data.

**Arguments**:

- `args` - Command-line arguments parsed by argparse.
  
  This function performs a series of data processing and analysis steps on the MSA data,
  including parsing, cleansing, dimension reduction, clustering, word cloud generation,
  feature selection, selecting residues, and generating logos.

<a id="protlearn.parse_args"></a>

#### parse\_args

```python
def parse_args()
```

Parse command-line arguments using argparse.

**Returns**:

- `Namespace` - An object containing parsed command-line arguments.

