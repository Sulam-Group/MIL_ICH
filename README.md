# _"Weakly-Supervised Learning Significantly Reduces the Number of Labels Required for Intracranial Hemorrhage Detection in Head CT"_

[![zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.7363182.svg)](https://zenodo.org/record/7363182)

This repository contains the source code to reproduce the experiments presented in the paper _"Weakly-Supervised Learning Significantly Reduces the Number of Labels Required for Intracranial Hemorrhage Detection in Head CT"_.

**DISCLAIMER: Due to the large size of the datasets used in this work, this repository does not contain the preprocesses images used for training. We refer the reader to the Methods section in the paper to download and preprocess the data.**

## Instructions to reproduce the figures in the paper

0. **Summary of the structure of this repository**

```bash
data    # to download from Zenodo
    ├── ...
models/ # to download from Zenodo
    ├── ...
notebooks/
    ├── examination_level_binary_classification
    │   ├── auc_label_complexity.ipynb          # Figure 6
    │   └── roc.ipynb                           # Figure 2
    ├── examination_level_hemorrhage_detection
    │   ├── ctich_tpr.ipynb                     # Figure 3.b, A2.b
    │   ├── rsna_f1_label_complexity.ipynb      # Figure 7, A3
    │   ├── rsna_f1_min_seq_l.ipynb             # Figure A1
    │   └── rsna_tpr.ipynb                      # Figure 3.a, A2.a
    └── image_level_hemorrhage_detection
    │   ├── cq500_explanations.ipynb            # Figure 4.a
    │   ├── cq500_f1.ipynb                      # Figure 5.a
    │   ├── ctich_explanations.ipynb            # Figure 4.b
    │   └── ctich_f1.ipynb                      # Figure 5.b
scripts/
    ├── explain
    │   └── ...
    └── predict
        └── ...
```

1. **Download the data**
The content of the `data` folder is available on Zenodo [https://zenodo.org/record/7363182](https://zenodo.org/record/7363182#.Y4Fpj-zMLdo). This folder contains the precomputed predictions of the models and the explanations of the predictions. The folder `data` should be placed in the root of this repository.

2. **Download the models**
The content of the `models` folder is available on Zenodo as well [https://zenodo.org/record/7363182](https://zenodo.org/record/7363182#.Y4Fpj-zMLdo). This folder contains the trained models. The folder `models` should be placed in the root of this repository. Note that it is not necessary to download the models to reproduce the figures, since the precomputed predictions are available in the `data` folder. To use the models and reproduce the predictions, it is necessary to download and preprocess the datasets as described in the Methods section of the paper, maintaining the splits used in this work. To predict and explain, use the scripts in the respective folders.

3. **Run the notebooks**
As noted in section `0`, run the notebooks to reproduce the corresponding figures.