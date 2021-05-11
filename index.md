# Modeling the Effects of Gene Perturbations in Maize

<p align="center">
  Contributors: Victor Avram, Sergio Jimenez, Eagon Meng, Wenhan Zhang
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/29682604/117849295-2bfd1200-b252-11eb-8d74-c186ab51c77e.png">
</p>

## Introduction and Motivation
---

As a plant breeding technology company, Inari's work comprises three high-level stages: (1) using computational methods to better understand biology, especially at the transcriptional level; (2) genetic editing of crops; and (3) delivery of altered genetic information to specific parts of the plant. By genetically modifying seeds, Inari seeks to increase crop diversity, increase yield so as to make efficient use of land, water, and fertilizer, and ultimately make socio-economic impacts by ensuring food security.

Our project sits within the first stage of Inari's work and seeks to answer "what are the effects of gene perturbations and are gene expression levels informative of one another?" Tackling these questions would provide Inari with a better understanding maize biology and would subsequently allow seed genetic editing with greater confidence. Essentially, we hope to document the relationships across genes, creating a network that can act as a look-up table to inform Inari of the genetic effects and side-effects of perturbing particular maize genes. These insights will help alleviate the risk of observing unexpected effects later in the breeding process. We have explored several methods to determine whether a subset (ideally a small subset) of genes can be used as predictors for the expression levels of the remaining genes in the maize genome. Through this process, we are able to realize relationships across genes and ultimately allow for more targeted gene modifications.

## Data and EDA
---

## Feature Selection and Regression-Based Methods
---

### Standard Feature Selection + Regression

### Landmark 1000 + Regression

## Autoencoder-Based Methods
---

### Autoencoder + Regression: Learning Structures of Expression Data

### Imputation using an Iterative Variational Autoencoder

Missing value imputation is widely used when analyzing gene expression data due to a phenomenon known as dropout. Either due to the detection threshold or inherent technical mishaps of the sequencer, the expression levels for certain genes for certain samples will not be registered. Instead of removing observations that contain missing values, often times missing value imputation is used in order to reconstruct these expression levels from the available information (i.e. the non-missing entries). Despite dropout being prevalent specifically in single-cell data (and we are using bulk RNA-seq data), various imputation models have been well-documented in the context of transcriptomics and therefore have led us to seek whether a missing value imputation framework could help us produce meaningful results.

Variational Autoencoders (VAE), similar to regular Autoencoders, can be used as tools for non-linear dimensionality reduction. Instead of the encoder mapping observations of an input **X** to points in a lower-dimensional latent space **Z**, the encoder maps observations of the input to associated latent distributions P(**Z**|**X**) (typically normal distributions paramaterized by the mean and standard deviation). The decoder samples from the laten distribution before reconstructing the input. The proposed method for iterative VAE is similar in concept to iterative principal component analysis (IPCA) used for imputation. Given missing values in a data matrix, IPCA iteratively performs PCA and updates the entries for the missing values until convergence criterium is met. The PCA module is replaced with a VAE network as seen in the figure below.
<p align="center">
  <img src="https://user-images.githubusercontent.com/29682604/117857931-556e6b80-b25b-11eb-8906-ef49176861aa.png">
</p>

Mean, k-nearest neighbors (kNN), and variational autoencoder (VAE) imputation models were built and compared. Each model was trained on the training set consisting of samples from 25 individual maize plants (n = 480). The training set did not contain any missing values. Trained models were used to impute missing values in the test set with samples from 1 individual maize plant (n = 21). The imputation models are described as follows.

**Performance:** The 3 imputation models were tested under two distinct scenarios. In scenario 1, a specified proportion of the entries in the expression matrix are selected and subsequently masked. In scenario 2, a specified subset of genes are selected and all entries corresponding to the selected subset of genes are subsequently masked. Scenario 2 is analogous to the case where a subset of genes are used as predictors in order to determine the expression levels for the response genes for which no data exists. Representations for scenario 1 and scenario 2 are given in Figure 9. Each model was tested under varying levels of missing data severity. For both scenarios, 5%, 10%, and 50% of the entries in the test set were masked, these entries posing as missing values. For every scenario-missing data severity pair, missing values were randomly generated 5 times. The average R2 values are reported in Table 4. The computation for the R2 values is based on the true values and imputed values for the missing data and does not take into account all entries in the test set as this would lead to an artificial inflation of the given model’s performance.


| Naive Mean R<sup>2</sup>   |   KNN R<sup>2</sup> |   VAE R<sup>2</sup> |
|:---------------:|:--------:|:--------:|
| 0.7231          | 0.9073   | 0.8985   |
| 0.7235          | 0.9075   | 0.9052   |
| 0.7223          | 0.8801   | 0.8936   |
| 0.7135          | 0.9070   | 0.9034   |
| 0.7206          | 0.9068   | 0.8981   |
| 0.7217          | 0.8923   | 0.8941   |


## Graph Neural Networks
---

## References

