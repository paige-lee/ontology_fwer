# ontology_fwer

## 1. Required files

#### 1.1  multilevel_lookup_table.txt
The original file of 289 structures. This is the raw data.

#### 1.2 multilevel2.csv
This file contains each of the 509 unique structures and the immediate parent and immediate child/children of each.

#### 1.3 adjacency_matrix2.csv
This file contains the adjacency matrix for all 509 unique structures. If the (i, j)th entry is 1, that means that the jth column is a child of the ith row. 

#### 1.4 subset.csv
This file contains the adjacency matrix for the eight subset structures. If the (i, j)th entry is 1, that means that the jth column is a child of the ith row. 

## 2. Creating ontology and subset

#### 2.1 Ontology code.ipynb
This Jupyter notebook contains the code for creating the brain ontology from the multilevel lookup table, yielding 509 unique structures in total. This notebook also contains the code for creating the subset of eight structures that we want to focus on. The file is very large, so you must download the zip file first to view it. 

#### 2.2 Subset - permutation testing summary.png
This image is a table that contains the hypothesis testing results of permutation testing (reject H0 or fail to reject H0) on the subset of eight structures.

#### 2.3 8 experiments.ipynb
This Jupyter notebook contains the code that generated 32,000 repeats of the parameter estimation and permutation testing algorithm. 4 cases * 8 experiments * 1000 repeats = 32,000 repeats in total. Each experiment varies a different single set of parameters so we can determine which parameter choices yield the best results.

## 3. False positive rate

#### 3.1 Calculating false positive rate (1000 permutations).ipynb
This Jupyter notebook contains code that calculates the false positive rate of each case/experiment combination for cases 1 and 2.

#### 3.2 False positive rate calculation table.pdf
This file is a table that contains the false positive rate calculations for each case/experiment combination for cases 1 and 2. It also contains the a table with the corresponding probabilities of observing estimated false positive rates at least this large, when the true false positive rate is 0.05.

## 4. False negative rate

#### 4.1 Calculating false negative rate (1000 permutations).ipynb
This Jupyter notebook contains code that calculates the false negative rate of each case/experiment combination for cases 2, 3, and 4.

#### 4.2 False negative rate calculation table.pdf
This file contains tables that present the false negative rates for all relevant structures in each case/experiment combination for cases 2, 3, and 4.

## 5. Defining functions

#### 5.1 Function definitions.ipynb
This Jupyter notebook contains all code for the items in sections 2, 3, and 4, where all code is compiled into functions.
