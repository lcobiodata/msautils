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
def __init__(msa_file, msa_format="fasta", *args, **kwargs)
```

Initialize the MSA object.

**Arguments**:

- `*args` - Variable-length positional arguments.
- `**kwargs` - Variable-length keyword arguments.

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
'Seq1/1-100': {21: 1, 25: 2, ...},
'Seq2/171-432': {101: 171, 103: 172, ...},
...
}

This mapping is useful for downstream analysis that requires knowing the position
of residues in the MSA.

**Notes**:

  - Residues represented by '-' (indels/gaps) are not included in the mapping.
  

**Example**:

  msa = MSA('example.fasta')
  msa.map_positions()
  
  Access the mapping:
  positions = msa.positions_map

<a id="protlearn.MSA.cleanse"></a>

#### cleanse

```python
def cleanse(indel='-',
            remove_lowercase=True,
            threshold=.9,
            plot=False,
            save=False,
            show=False)
```

Cleanse the MSA data by removing columns and rows with gaps/indels.

**Arguments**:

- `indel` _str, optional_ - The character representing gaps/indels (default is '-').
- `remove_lowercase` _bool, optional_ - Whether to remove lowercase characters (default is True).
- `threshold` _float, optional_ - The threshold for gaps/indels (default is 0.9).
- `plot` _bool, optional_ - Whether to plot a heatmap of gaps/indels (default is False).
- `save` _bool, optional_ - Whether to save a heatmap of gaps/indels (default is False).
- `show` _bool, optional_ - Whether to show a heatmap of gaps/indels (default is False).

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

  msa = MSA('example.fasta')
  msa.map_positions()
  msa.cleanse()
  msa.reduce(plot=True)

<a id="protlearn.MSA.cluster"></a>

#### cluster\_sequences

```python
def cluster(min_clusters=2,
                      max_clusters=10,
                      method='single-linkage')
```

Cluster the MSA data and obtain cluster labels.

**Arguments**:

- `min_clusters` _int, optional_ - Minimum number of clusters (default is 2).
- `max_clusters` _int, optional_ - Maximum number of clusters (default is 10).
- `method` _str, optional_ - Clustering method ('k-means' or 'single-linkage') (default is 'single-linkage').
  

**Notes**:

  - This method performs clustering on the MSA data and assigns cluster labels to sequences.
  - Clustering can be done using either k-means or single-linkage methods.
  - The optimal number of clusters is determined using silhouette scores.
  - The cluster labels are stored in the 'labels' attribute of the MSA object.
  

**Example**:

  msa = MSA('example.fasta')
  msa.map_positions()
  msa.cleanse()
  msa.cluster(method='single-linkage', min_clusters=3, plot=True)

<a id="protlearn.MSA.generate_wordclouds"></a>

#### generate\_wordclouds

```python
def generate_wordclouds(metadata=None,
                        column='Protein names',
                        plot=False,
                        save=False,
                        show=False)
```

Generate word cloud visualizations from protein names in a DataFrame.

**Arguments**:

- `metadata` _str, default=None_ - Path to metadata file in tsv format.
- `column` _str, default='Protein names'_ - The name of the column in the DataFrame containing protein names.
- `plot` _bool, default=False_ - Whether to plot word clouds.
- `save` _bool, default=False_ - Whether to save word clouds.
- `show` _bool, default=False_ - Whether to show word clouds.
  
  This method extracts substrate and enzyme names from the specified column using regular expressions,
  normalizes the names, and creates word cloud plots for each cluster of sequences.
  

**Example**:

  msa = MSA('example.fasta')
  msa.map_positions()
  msa.cleanse()
  msa.cluster(method='single-linkage', min_clusters=3)
  msa.generate_wordclouds(metadata='metadata.tsv', plot=True)

<a id="protlearn.MSA.select_features"></a>

#### select\_features

```python
def select_features(n_estimators=None,
                    random_state=None,
                    plot=False,
                    save=False,
                    show=False)
```

Select important features (residues) from the MSA data.

**Arguments**:

- `n_estimators` _int, optional_ - Parameter n_estimators for RandomForest.
- `random_state` - (int, optional): Parameter random_state for RandomForest.
- `plot` _bool, optional_ - Whether to plot feature selection results (default is False).
- `save` _bool, optional_ - Whether to save feature selection results (default is False).
- `show` _bool, optional_ - Whether to show feature selection results (default is False).

<a id="protlearn.MSA.select_residues"></a>

#### select\_residues

```python
def select_residues(threshold=0.9,
                    top_n=None,
                    plot=False,
                    save=False,
                    show=False)
```

Select and store residues to be candidates for Specificity-determining Positions (SDPs) from the MSA data.

**Arguments**:

- `threshold` _float, optional_ - The threshold for selecting residues based on importance (default is 0.9).
- `top_n` _int, optional_ - The top N residues to select based on importance (default is None).
- `plot` _bool, optional_ - Whether to plot the selected residues (default is False).
- `save` _bool, optional_ - Whether to save the selected residues (default is False).
- `show` _bool, optional_ - Whether to show the selected residues (default is False).

<a id="protlearn.MSA.generate_logos"></a>

#### generate\_logos

```python
def generate_logos(plot=False, save=False, show=False)
```

Generate logos from the MSA data for each cluster label.

**Arguments**:

- `plot` _bool, optional_ - Whether to plot the generated logos (default is False).
- `save` _bool, optional_ - Whether to save the generated logos (default is False).
- `show` _bool, optional_ - Whether to show the generated logos (default is False).

<a id="protlearn.main"></a>

#### main

```python
def main()
```

Main function to process the Multiple Sequence Alignment (MSA) data.

This function performs a series of data processing and analysis steps on the MSA data,
including parsing, cleansing, dimension reduction, clustering, word cloud generation,
feature selection, selecting residues, and generating logos.

