from fastai.tabular import *
import torch
import torchvision

path = "./datasets"
dataset_file = "./datasets/heart.csv"
df = pd.read_csv(dataset_file)
df.head()

selected_df_split_size = (int)(len(df) / 2)
procs = [FillMissing, Categorify, Normalize]
valid_idx = range(selected_df_split_size, len(df))

# age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
dep_var = "target"
cat_names = ["age", "thal"]

data = TabularDataBunch.from_df(path, df, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)
print(data.train_ds.cont_names)

#(cat_x, cont_x), y = next(iter(data.train_dl))
# for o in (cat_x, cont_x, y): print(to_np(o[:5]))

learner = tabular_learner(data, layers=[200, 100], emb_szs={"native-country": 10}, metrics=accuracy)
learner.fit_one_cycle(1, 1e-2)

predict_test = {
    "age": 60,
    "sex": 1,
    "cp": 4,
    "trestbps": 130,
    "chol": 206,
    "fbs": 0,
    "restecg": 2,
    "thalach": 150,
    "exang": 1,
    "oldpeak": 2.3,
    "slope": 2,
    "ca": 2,
    "thal": "reversible",
    "target": -1,
}

print(predict_test)
print(learner.predict(predict_test))

traced_script_module = torch.jit.trace(learner.predict, predict_test)
traced_script_module.save("./model.pt")
