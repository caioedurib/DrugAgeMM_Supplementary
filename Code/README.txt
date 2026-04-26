Author: Caio Ribeiro
Contact: cer29@kent.ac.uk

## Overview

This script implements a machine learning pipeline for classification tasks using a Random Forest model. It includes data preprocessing, custom stratified cross-validation, feature importance analysis, and false positive analysis.

The pipeline is designed to work with datasets containing compound-level data, ensuring that related entries (e.g., same compound) are not split across training and testing sets.

Experiments ran on Python 3.9
This will likely work fine on later versions of Python, but this has not been tested.

---

## Features

* Custom 5-fold cross-validation that prevents data leakage by grouping compounds
* Random Forest classifier with balanced class weighting
* Performance metrics:

  * True Positives (TP)
  * False Positives (FP)
  * True Negatives (TN)
  * False Negatives (FN)
  * ROC AUC
  * Precision-Recall AUC
* Feature importance extraction
* False positive analysis with prediction probabilities
* Dataset preprocessing:

  * Removal of low-frequency features
  * Removal of invalid instances
  * Optional filtering by sex (M/F)

---

## Requirements

Install the following Python libraries before running:

* pandas
* numpy
* scikit-learn

You can install them via pip:

```
pip install pandas numpy scikit-learn
```

---

## Input Data Format

The input dataset must be a tab-separated (.tsv) file with:

* An index column
* A column named `Compound_name` (unique identifier for compounds)
* A column named `class` (binary: 0 or 1)
* A column named `sex` (values: 'M' or 'F')
* Remaining columns should be binary features (0/1)

---

## Usage

Run the script from the command line:

```
python main.py -i <input_file> -o <output_file> [options]
```

### Required Arguments

* `-i`, `--input`
  Path to the input dataset file

* `-o`, `--output`
  Path to the output results file

### Optional Arguments

* `-s`, `--sex`
  Filter dataset by sex
  Options: `all` (default), `M`, `F`

* `-r`, `--random-seed`
  Random seed for reproducibility (default: 101)

* `-c`, `--cross-validation`
  Run cross-validation

* `-f`, `--feature-importance`
  Run feature importance analysis

* `-a`, `--false-positive`
  Run false positive analysis

⚠️ If no analysis flags are provided, all analyses will run by default.

---

## Example

Run full pipeline:

```
python main.py -i data.tsv -o results.txt
```

Run only cross-validation:

```
python main.py -i data.tsv -o results.txt -c
```

Run feature importance and false positive analysis:

```
python main.py -i data.tsv -o results.txt -f -a
```

---

## Output

The output file will contain:

1. **Cross-validation results**

   * Metrics for each fold
   * Summary statistics

2. **Feature importance**

   * Ranked list of features with importance scores

3. **False positive analysis**

   * Prediction probabilities for each instance
   * Fold assignment and metadata

---

## Notes

* The script uses a custom fold assignment strategy to balance class distributions.
* Compounds are kept intact within folds to avoid data leakage.
* Features must be binary (0/1); non-binary features will be skipped with warnings.
* Instances with no active features are removed during preprocessing.