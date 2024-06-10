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
