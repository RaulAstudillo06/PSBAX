#%%
import os
import sys
import torch
import pandas as pd
import numpy as np

torch.set_default_dtype(torch.float64)
def seed_torch(seed, verbose=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_torch(0)
#%%

problem_lst = [
    "schmidt_2021_ifng", # 17465
    "schmidt_2021_il2", # 17465
    "zhuang_2019", # 17528
    "sanchez_2021_tau", # 17176
    "zhu_2021_sarscov2_host_factors", # 16670
]

problem = problem_lst[1]
random = False
data_size = 5000


data_path = f"./{problem}.csv"
df = pd.read_csv(data_path, index_col=0)
print(df.shape)
#%%
df_y = df["y"]
def topk_indices(y: pd.Series, k):
    if isinstance(k, float):
        return list(y.sort_values(ascending=False).index[:int(k * y.shape[0])].values)
    elif isinstance(k, int):
        return list(y.sort_values(ascending=False).index[:k].values)
if random:
    keep_idx = np.random.choice(df_y.index, data_size, replace=False)
else:
    keep_idx = topk_indices(df_y, data_size)

df = df.loc[keep_idx]

if random:
    df.to_csv(f"{problem}_random_{data_size}.csv")
else:
    df.to_csv(f"{problem}_top_{data_size}.csv")



#%%
