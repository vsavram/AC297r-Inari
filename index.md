# Modeling the Effects of Gene Perturbations in Maize

<p align="center">
  Harvard Institute for Applied Computational Science
</p>
<p align="center">
  Contributors: Victor Avram, Sergio Jimenez, Eagon Meng, Wenhan Zhang
</p>
<p align="center">
    IACS Faculty: Chris Tanner and Phoebe Wong
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

The data is comprised of gene expression values derived from 26 individual maize plants. Multiple samples were taken from 10 different tissues from these 26 individuals and subsequently sequenced in order to obtain gene expression counts. The number of samples contributed varies per individual and varies per tissue. As well, samples were collected at different developmental stages. The samples collected from 1 of the individuals are set aside for validation (referred to as the test set), leaving the samples collected from the remaining 25 individuals as the data to be used for model creation and model improvement (referred to as the training set). These datasets are provided in tabular form. Each entry represents the quantification of the expression level for the given gene in the given sample. Therefore, the set of genes analyzed is consistent across all samples. The training set contains 480 samples, each with expression levels for 46,430 genes. The test set contains 21 samples, each with expression levels for the same 46430 genes found in the training set.

The raw gene expression counts are first converted to transcripts per million (TPM). The steps of this preprocessing technique are as follows: 1) Divide the expression level by the length of the given gene in kilobases, 2) Divide by the summation of gene length normalized expression levels in the given sample, 3) Multiply by 1,000,000. The process of converting raw counts to TPM first normalized the expression levels by the gene length. More fragments are likely to map to longer genes and vice versa for shorter genes. These gene length normalized expression levels are then adjusted for sequencing depth. Sequencing depth refers to the total counts attributed to a given sample. Given the inability to sequence all samples at the same time and with the same machinery, as well as the possibility that different protocols were used for different samples, sequencing depth normalization is an important step in being able to make comparisons across samples. Lastly, the expression levels are scaled by a large factor.

## Feature Selection and Regression-Based Methods
---

### Standard Feature Selection + Regression

### Landmark 1000 + Regression

## Autoencoder-Based Methods
---

### Autoencoder + Regression: Learning Structures of Expression Data

Can we learn some general features of a subset of the data that are informative of another subset of the data? Using this leading question, we implemented an autoencoder neural network architecture to discern whether non-linear dimensionality reduction paired with linear regression would be robust in terms of our predictive task. Autoencoders are neural networks with a two-part architecture consisting of an encoder and decoder. The encoder takes an input **X** and maps the input to its latent representation **Z** (often times the latent embedding is lower-dimensional relative to the input). The decoder takes the latent representation and aims to create a reconstructed version of the input **X'**. 

The autoencoder + regression model consists of the following steps. First, an autoencoder is trained on a subset of genes which we can term as our predictor set (or predictor genes). Subsequently, the learned latent embedding of the predictor set is used for the multiple linear regression task of predicting each response gene independently. Given *n* response genes, we need to train *n* multiple linear regression models. 

![Screen Shot 2021-05-14 at 10 32 57 AM](https://user-images.githubusercontent.com/29682604/118285744-c605ca00-b49f-11eb-9892-7d911f778ec4.png)

**Performance:** Unfortunately, the autoencoder + regression model exhibited a much lower performance compared to a vanilla neural network with dropout layers. Even after altering the depth of the autoencoder and the dimensionality of the embedding space (i.e. the width of the bottleneck), the autoencoder + regression model lacked substantial performance gains. We can interpret these results in the following ways. It may be better for a given model to be privy to all of the data, despite the potential of capturing noise instead of true signal. Furthermore, dropout may be a more effective means of regularization as opposed to dimensionality reductions in this context. 

### Imputation using an Iterative Variational Autoencoder

Missing value imputation is widely used when analyzing gene expression data due to a phenomenon known as dropout. Either due to the detection threshold or inherent technical mishaps of the sequencer, the expression levels for certain genes for certain samples will not be registered. Instead of removing observations that contain missing values, often times missing value imputation is used in order to reconstruct these expression levels from the available information (i.e. the non-missing entries). Despite dropout being prevalent specifically in single-cell data (and we are using bulk RNA-seq data), various imputation models have been well-documented in the context of transcriptomics and therefore have led us to seek whether a missing value imputation framework could help us produce meaningful results.

Variational Autoencoders (VAE), similar to regular Autoencoders, can be used as tools for non-linear dimensionality reduction. Instead of the encoder mapping observations of an input **X** to points in a lower-dimensional latent space **Z**, the encoder maps observations of the input to associated latent distributions P(**Z**|**X**) (typically normal distributions paramaterized by the mean and standard deviation). The decoder samples from the laten distribution before reconstructing the input. The proposed method for iterative VAE is similar in concept to iterative principal component analysis (IPCA) used for imputation. Given missing values in a data matrix, IPCA iteratively performs PCA and updates the entries for the missing values until convergence criterium is met. The PCA module is replaced with a VAE network as seen in the figure below. At each iteration, non-linear dimensionality reduction is performed on the data matrix and the reconstucted values are used to update the missing value entries.
<p align="center">
  <img src="https://user-images.githubusercontent.com/29682604/117857931-556e6b80-b25b-11eb-8906-ef49176861aa.png">
</p>

Mean, k-nearest neighbors (kNN), and variational autoencoder (VAE) imputation models were built and compared. Each model was trained on the training set consisting of samples from 25 individual maize plants (n = 480). The training set did not contain any missing values. Trained models were used to impute missing values in the test set with samples from 1 individual maize plant (n = 21). The imputation models are described as follows.

**Performance:** The 3 imputation models were tested under two distinct scenarios. In scenario 1, a specified proportion of the entries in the expression matrix are selected and subsequently masked. In scenario 2, a specified subset of genes are selected and all entries corresponding to the selected subset of genes are subsequently masked. Scenario 2 is analogous to the case where a subset of genes are used as predictors in order to determine the expression levels for the response genes for which no data exists. Representations for scenario 1 and scenario 2 are given in Figure 9. Each model was tested under varying levels of missing data severity. For both scenarios, 5%, 10%, and 50% of the entries in the test set were masked, these entries posing as missing values. For every scenario-missing data severity pair, missing values were randomly generated 5 times. The average R2 values are reported in Table 4. The computation for the R2 values is based on the true values and imputed values for the missing data and does not take into account all entries in the test set as this would lead to an artificial inflation of the given model’s performance.


| --- | Naive Mean R<sup>2</sup> |   KNN R<sup>2</sup> | VAE R<sup>2</sup> |
|:---------------:|:---------------:|:--------:|:--------:|
Scenario 1: 5% of entries missing | 0.7231 | 0.9073   | 0.8985   |
Scenario 1: 10% of entries missing | 0.7235 | 0.9075   | 0.9052   |
Scenario 1: 50% of entries missing | 0.7223 | 0.8801   | 0.8936   |
Scenario 2: 5% of entries missing | 0.7135 | 0.9070   | 0.9034   |
Scenario 2: 10% of entries missing | 0.7206 | 0.9068   | 0.8981   |
Scenario 2: 50% of entries missing | 0.7217 | 0.8923   | 0.8941   |


## Graph Neural Networks
---

## References

