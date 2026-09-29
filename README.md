# GenProt-DSM

GenProt-DSM is a cross-modal deep learning framework for pathogenicity prediction of brain-disorder-associated missense variants. It integrates DNA-level evolutionary constraints derived from GPN-MSA with protein-level sequence semantics derived from ProtT5-XL, followed by sequence encoding, gated multimodal fusion, and downstream pathogenicity prediction.

## Related Paper

**GenProt-DSM: Integrating Evolutionary Constraints and Protein Semantics with Pretrained Language Models for Missense Variant Pathogenicity Prediction in Brain Disorders**

## Repository Overview

This repository provides the downstream training, evaluation, statistical analysis, and visualization code used in GenProt-DSM.

The main analyses include:

- GenProt-DSM training and evaluation on the gene-disjoint benchmark
- evaluation of general-purpose pathogenicity predictors
- comparison between brain-disorder-specific and general-disease training
- comparison with disease-specific pathogenicity predictors
- disease-stratified performance analysis
- rare-variant evaluation
- paired statistical comparisons
- calibration and decision-curve analysis
- ablation experiments
- fusion-strategy experiments
- gene-level distribution analysis
- manuscript and supplementary figure generation

## Repository Structure

```text
GenProt-DSM/
├── README.md
├── requirements.txt
└── src/
    ├── model.py
    ├── ablation.py
    ├── calibration_analysis.py
    ├── disease_bootstrap.py
    ├── fusion.py
    ├── gene_distribution.py
    ├── paired_statistical_test.py
    ├── rare_gene_tools.py
    ├── tool_metrics.py
    └── visualization/
        ├── plot_Fig1_revised_dataset.py
        ├── plot_Fig3_model_selection.py
        ├── plot_Fig4_general_predictors.py
        ├── plot_Fig5_disease_specific_robustness.py
        ├── plot_Fig6_ablation_umap.py
        ├── plot_Fig7_SHAP.py
        ├── plot_Fig8_additional_prediction_analyses.py
        ├── plot_Supplementary_FigS1_gene_distribution.py
        ├── plot_Supplementary_FigS2_tool_missingness.py
        └── plot_Supplementary_FigS3_calibration_DCA.py
```

## Dataset

The primary benchmark uses a gene-disjoint training/test split.

### VarGeneDisjointTrain

- 3,172 variants
- 1,586 pathogenic variants
- 1,586 benign variants
- 1,618 unique genes

### VarGeneDisjointTest

- 786 variants
- 393 pathogenic variants
- 393 benign variants
- 402 unique genes

There is no gene overlap between VarGeneDisjointTrain and VarGeneDisjointTest.

To further reduce potential information leakage caused by sequence similarity between different genes, a train-test protein sequence-similarity audit was performed using MMseqs2. A train-test protein pair was considered a similarity violation when:

- sequence identity >= 30%
- query coverage (qcov) >= 80%
- target coverage (tcov) >= 80%

No train-test protein pair satisfied this predefined criterion. Among alignments satisfying qcov >= 80% and tcov >= 80%, the maximum sequence identity was 29.8%.

## Public Data and Feature Files

The processed dataset and pre-extracted feature files used for the downstream experiments are publicly available at:

https://www.kaggle.com/datasets/yufei4216/genprot-dsm-data

The current Kaggle dataset contains the following files:

```text
1test_gpn_alt.npy
1test_gpn_ref.npy
1test_t5_mut.npy
1test_t5_wt.npy

1train_gpn_alt.npy
1train_gpn_ref.npy
1train_t5_mut.npy
1train_t5_wt.npy

General_train.csv
VarGeneDisjointTest.csv
VarGeneDisjointTrain.csv

train_gpn_alt.npy
train_gpn_ref.npy
train_t5_mut.npy
train_t5_wt.npy
```

### File Naming Convention

The `1train_*` and `1test_*` feature arrays correspond to the brain-disorder-specific GenProt-DSM experiment.

The following files:

- `1train_gpn_ref.npy`
- `1train_gpn_alt.npy`
- `1train_t5_wt.npy`
- `1train_t5_mut.npy`

are the training features corresponding to `VarGeneDisjointTrain.csv`.

The following files:

- `1test_gpn_ref.npy`
- `1test_gpn_alt.npy`
- `1test_t5_wt.npy`
- `1test_t5_mut.npy`

are the test features corresponding to `VarGeneDisjointTest.csv`.

The feature arrays without the `1` prefix are used for the GenProt-DSM-General training experiment:

- `train_gpn_ref.npy`
- `train_gpn_alt.npy`
- `train_t5_wt.npy`
- `train_t5_mut.npy`

and correspond to `General_train.csv`.

GenProt-DSM-General is evaluated on the same `VarGeneDisjointTest.csv` used for the brain-disorder-specific model. Therefore, the same `1test_*` feature arrays are used for final testing.

## Expected Local Data Layout

After downloading the Kaggle dataset, the files can be organized locally as follows:

```text
data/
├── VarGeneDisjointTrain.csv
├── VarGeneDisjointTest.csv
├── General_train.csv
├── 1train_gpn_ref.npy
├── 1train_gpn_alt.npy
├── 1train_t5_wt.npy
├── 1train_t5_mut.npy
├── 1test_gpn_ref.npy
├── 1test_gpn_alt.npy
├── 1test_t5_wt.npy
├── 1test_t5_mut.npy
├── train_gpn_ref.npy
├── train_gpn_alt.npy
├── train_t5_wt.npy
└── train_t5_mut.npy
```

Local paths in the scripts may need to be updated according to the user's environment.

## Pretrained Models

GenProt-DSM uses representations derived from two pretrained biological language models:

- **GPN-MSA** for DNA-level evolutionary constraints
- **ProtT5-XL** for protein-level sequence semantics

This repository does not redistribute the original pretrained model code or pretrained model weights.

The public Kaggle dataset provides the pre-extracted feature arrays required for the downstream GenProt-DSM experiments. Therefore, reproduction of the downstream analyses does not require rerunning the upstream GPN-MSA or ProtT5-XL feature-extraction procedure.

## Environment

The downstream code was tested with Python 3.10 on Windows.

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## Main Scripts

### Main Model Training and Evaluation

Train and evaluate GenProt-DSM:

```bash
python src/model.py
```

For the brain-disorder-specific experiment, use:

- `VarGeneDisjointTrain.csv`
- `VarGeneDisjointTest.csv`
- the corresponding `1train_*` and `1test_*` feature arrays

### General-Purpose Predictor Evaluation

Evaluate pathogenicity-prediction tools and calculate performance metrics:

```bash
python src/tool_metrics.py
```

### Ablation Analysis

Run feature- and module-ablation experiments:

```bash
python src/ablation.py
```

### Fusion-Strategy Analysis

Evaluate multimodal fusion strategies:

```bash
python src/fusion.py
```

### Disease-Stratified Evaluation

Run disease-stratified evaluation and bootstrap confidence-interval analysis:

```bash
python src/disease_bootstrap.py
```

### Rare-Variant Analysis

Run rare-variant evaluation:

```bash
python src/rare_gene_tools.py
```

### Paired Statistical Comparisons

Perform paired statistical comparisons between GenProt-DSM and selected baseline predictors:

```bash
python src/paired_statistical_test.py
```

AUROC differences are evaluated using DeLong's test, whereas AUPR and MCC differences are evaluated using paired bootstrap resampling.

### Calibration and Decision-Curve Analysis

Run calibration and decision-curve analysis:

```bash
python src/calibration_analysis.py
```

### Gene-Level Distribution Analysis

Analyze gene-level variant distributions:

```bash
python src/gene_distribution.py
```

## Brain-Disorder-Specific vs. General-Disease Training

To evaluate whether brain-disorder-specific training contributed to predictive performance, GenProt-DSM was compared with a control model, GenProt-DSM-General.

GenProt-DSM-General used:

- the same model architecture
- the same hyperparameter configuration
- the same training strategy
- the same sample size and class distribution
- the same final evaluation set

The general-disease training set is provided as:

```text
General_train.csv
```

The corresponding pre-extracted training features are:

```text
train_gpn_ref.npy
train_gpn_alt.npy
train_t5_wt.npy
train_t5_mut.npy
```

The pathogenic variants in the general-disease training set cover:

- cancer
- cardiovascular diseases
- endocrine diseases
- kidney diseases

Genes represented in `VarGeneDisjointTest.csv` were excluded from GenProt-DSM-General training to prevent gene-level leakage.

Both GenProt-DSM and GenProt-DSM-General were evaluated on the same `VarGeneDisjointTest.csv` using the same `1test_*` feature arrays.

## Evaluation Design

All architecture and hyperparameter selection was performed exclusively using training/validation data within VarGeneDisjointTrain.

Five-fold gene-grouped cross-validation was used so that variants from the same gene were not distributed across the corresponding training and validation folds.

VarGeneDisjointTest remained untouched during architecture selection, hyperparameter tuning, and threshold selection and was used only for final evaluation.

For final evaluation, the output probabilities of the five fold-specific models were averaged.

## Rare-Variant Evaluation

Rare variants were defined as variants with allele frequency:

```text
AF < 0.01
```

Variants with missing allele-frequency annotations were excluded.

The final rare-variant subset contained:

- 177 variants
- 164 pathogenic variants
- 13 benign variants

Because this subset is strongly imbalanced toward pathogenic variants, AUPR should be interpreted together with AUROC and MCC rather than in isolation.

## Calibration and Decision Analysis

Calibration was evaluated using:

- Brier score
- expected calibration error (ECE)
- calibration intercept
- calibration slope

Decision-curve analysis was used to evaluate threshold-dependent benchmark-level net benefit.

The predefined GenProt-DSM ensemble decision threshold was:

```text
0.5544
```

This threshold was determined from training/validation data and was not optimized on VarGeneDisjointTest.

Because VarGeneDisjointTest is a balanced benchmark dataset rather than a population-representative clinical cohort, decision-curve results should be interpreted as benchmark-level decision utility rather than direct estimates of population-level clinical benefit.

## Visualization

Scripts used to generate manuscript and supplementary figures are provided under:

```text
src/visualization/
```

The visualization scripts currently included in the repository are:

- `plot_Fig1_revised_dataset.py`
- `plot_Fig3_model_selection.py`
- `plot_Fig4_general_predictors.py`
- `plot_Fig5_disease_specific_robustness.py`
- `plot_Fig6_ablation_umap.py`
- `plot_Fig7_SHAP.py`
- `plot_Fig8_additional_prediction_analyses.py`
- `plot_Supplementary_FigS1_gene_distribution.py`
- `plot_Supplementary_FigS2_tool_missingness.py`
- `plot_Supplementary_FigS3_calibration_DCA.py`

These scripts reproduce dataset-characterization, model-selection, baseline-comparison, robustness, ablation, UMAP, SHAP, additional prediction, gene-distribution, prediction-missingness, and calibration/DCA visualizations used in the manuscript and supplementary materials.

## Reproducibility

A typical downstream workflow is:

1. Download the processed datasets and pre-extracted feature arrays from Kaggle.
2. Organize the files according to the expected local directory structure.
3. Install the required Python dependencies.
4. Update local file paths in the scripts if necessary.
5. Run `src/model.py` for the primary GenProt-DSM experiment.
6. Run `src/tool_metrics.py` for general-purpose predictor evaluation.
7. Run `src/ablation.py` and `src/fusion.py` for model-component analyses.
8. Run `src/disease_bootstrap.py` for disease-stratified evaluation.
9. Run `src/rare_gene_tools.py` for rare-variant analysis.
10. Run `src/paired_statistical_test.py` for paired statistical comparisons.
11. Run `src/calibration_analysis.py` for calibration and decision-curve analysis.
12. Run the scripts under `src/visualization/` to reproduce manuscript and supplementary figures.

## Data Source Statement

Pathogenic missense variants were curated from:

- HGMD Professional
- ClinVar

Benign missense variants were obtained from:

- ClinVar

All variants were restricted to single-nucleotide variants (SNVs), mapped to the hg38 reference genome, and processed according to the procedures described in the manuscript.

HGMD Professional is a licensed resource. Users should comply with all applicable HGMD licensing and redistribution requirements.

Only data and derived files permitted for redistribution should be included in publicly distributed repositories or datasets.

## Data and Code Availability

The GenProt-DSM source code is publicly available at:

https://github.com/Lilab-genomics/GenProt-DSM

The processed dataset and pre-extracted feature files used for downstream analyses are publicly available at:

https://www.kaggle.com/datasets/yufei4216/genprot-dsm-data

## Citation

If you use GenProt-DSM, the accompanying source code, or the processed feature data, please cite:

> **GenProt-DSM: Integrating Evolutionary Constraints and Protein Semantics with Pretrained Language Models for Missense Variant Pathogenicity Prediction in Brain Disorders**

A complete bibliographic citation will be added after publication.

## Contact

For questions regarding the code, data, or implementation of GenProt-DSM, please contact the corresponding author.
