# Semi-supervised deep transfer learning accurately predicting single-cell drug responses

## Introduction

**SSDA4Drug** is a semi-supervised domain adaptation model for inferring single-cell drug response.

## Model architecture

![](framework.png)

## Requirements

The deep learning models were trained on 2*NVIDIA GeForce RTX 4090 on linux.

+ Python 3.9
+ PyTorch 2.1.0
+ Pandas 2.0.3
+ Numpy 1.24.3
+ Scikit-learn 1.3.1
+ Scanpy 1.9.5
+ Shap 0.45.1
+ Captum 0.7.0

## Usage

To setup the environment, install conda and run (Must run on servers with multiple GPUs):

```bash
conda create --name <your_env_name> --file requirements.txt
```

If you want to use SCAD data to train a model, please splite the dataset
```bash
SCAD_Drug.ipynb
split_data.ipynb
```
then run
```bash
python SSDA.py -shot_method 3-shot -drug Gefitinib
```

if you want to train the model using other datasets, you need to download the datasets (Example: GDSC and GEO), place it in the datasets folder, and then run
```bash
python SSDA.py -shot_method 3-shot
```

## Directory structure
+ `SSDA.py/experiment_shot.py`: contains the code for the model, the dataset, the evaluation, and the training loop.

+ `trainer.py`: Contains the training loop, the hyperparameters, and the evaluation.

+ `utils.py`: Contains auxiliary, general-purpose, or custom functions, which can be called and used in other parts of the project.

+ `model.py`: Model storage directory.

+ `data`:
  * SCAD_data：The directory stores the dataset of SCAD.
  * experiment_data: The directory stores the dataset of experiment.
