# Fastai/Pytorch Model Workflow

* Setting up the environment via conda
* Table to dataset in minutes via feature_columns -> input vector
* Model compilation
* Model torchscript conversion
* Model export
* C++ serving scalable model deployment

## Setup

```bash
brew install python3
open https://www.anaconda.com/distribution/#macos
# download the installer for python 3.7 and install it
nano ~/.zshrc
export PATH=/anaconda3/bin:$PATH
# restart shell
conda init zsh
# restart shell
```

## Creating the environment

```bash
cd pt-model-workflow
conda create -n ptmw python=3.6 pip
conda activate ptmw
conda install -c pytorch -c fastai fastai
# conda deactivate # in case of leaving
```

## Compiling, Training, Tuning and Exporting the model

```bash
python train.py
```