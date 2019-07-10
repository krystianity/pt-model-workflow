from fastai.tabular import *
import torch
import torchvision
import pandas as pd
import numpy as np
from aiprocs import aip

path = "./datasets"
dataset_file = "./datasets/heart.csv"
df = pd.read_csv(dataset_file)
df.head()

train_df, valid_df, test_df = aip.split_df(df, 0.2, 0.05)

# age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
options = aip.create_options(
    ["FillMissing", "Normalize", "Categorify"], ["age", "thal"], "target", aip.MODEL_TYPES.PYTORCH.value
)

metadata = aip.analyse_df(train_df, options)
train_df, valid_df, test_df = aip.prepare_dfl([train_df, valid_df, test_df], metadata)

# fastai requires a combined df with validation indices
train_valid_df = pd.concat([train_df, valid_df])
valid_idx = range(len(train_valid_df)-len(valid_df), len(train_valid_df))

# we still pass the categories to fastai, because we want it to create embeddings for us
data = TabularDataBunch.from_df(path, train_valid_df, options["dep_var"], bs=2, valid_idx= valid_idx,
    procs=[Categorify], cat_names=options["cat_names"], test_df=test_df)
learner = tabular_learner(data, layers=[200, 100], emb_szs={"native-country": 10}, metrics=accuracy)
learner.fit_one_cycle(1, 1e-2)

# extract real input tensor for jit tracing
extraced_loader_input, _ = next(iter(learner.data.valid_dl))
print(extraced_loader_input)
# trace model and export
traced_script_model = torch.jit.trace(learner.model, extraced_loader_input)
torch.jit.save(traced_script_model, "./model.pt")

# export metadata to json file
aip.store_metadata(metadata, "./metadata.json")