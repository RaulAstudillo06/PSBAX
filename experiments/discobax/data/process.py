# load "achilles.h5" and "sanchez_2021_neruons_tau.h5"

#%%
import sys
import numpy as np
import pandas as pd
import os
import pickle as pkl
from collections import defaultdict
import torch

import slingpy as sp
from slingpy.data_access.data_sources.hdf5_tools import HDF5Tools
from slingpy.data_access.data_sources.hdf5_data_source import HDF5DataSource
from slingpy.data_access.data_sources.abstract_data_source import AbstractDataSource

from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# dataset_x should be the same as the dataset that was used for clustering?
if "discobax" not in os.getcwd():
    # change the path to the location of the achilles.h5 file
    os.chdir("experiments/data/discobax")

print(os.getcwd())

def get_topk_indices(
        dataset_y: AbstractDataSource, topk_percent: float = 0.1
    ):
    y_indices = dataset_y.get_row_names()
    y_values = pd.DataFrame(
        dataset_y.get_data()[0].flatten(), index=y_indices, columns=["y_values"]
    )
    y_values.sort_values(by=["y_values"], ascending=False, axis=0, inplace=True)
    num_k = int(topk_percent * len(y_values.y_values))
    topk_indices = list(y_values.iloc[:num_k].index) # 1717 = 0.1 * 17176
    return topk_indices

def find_optimal_number_clusters(dataset_x, min_components=2, max_components=100, num_repeats=10, plot_location="temp.png"):
    n_components = np.arange(min_components, max_components)
    models = [GMM(n, covariance_type='full', random_state=0, max_iter=200).fit(dataset_x) for n in n_components for repeat in range(num_repeats)]
    bic = np.array([m.bic(dataset_x) for m in models]).reshape(len(n_components),num_repeats).mean(axis=1).flatten()
    aic = np.array([m.aic(dataset_x) for m in models]).reshape(len(n_components),num_repeats).mean(axis=1).flatten()
    # plt.plot(n_components, bic, label='BIC')
    # plt.plot(n_components, aic, label='AIC')
    argmin_AIC = min_components + np.argmin(np.array(aic))
    # plt.axvline(x = argmin_AIC, color = 'r', label = 'Opt. # clus: {}'.format(argmin_AIC), linestyle='dashed')
    # plt.legend(loc='best')
    # plt.xlabel('n_components')
    # plt.savefig(plot_location)
    # plt.title('AIC & BIC Vs number of GMM components')
    return argmin_AIC

def get_top_target_clusters(
        dataset_x, 
        topk_indices, 
        num_clusters=20, 
        n_components=20,
        plot_location="temp.png",
        path_cache_cluster_files_prefix="clusters"
    ):
    dataset_x = dataset_x.subset(topk_indices)
    row_names = dataset_x.get_row_names()
    x = dataset_x.get_data()[0]
    dict_index_to_item = {}
    for index,item in enumerate(row_names):
        dict_index_to_item[index] = item
    #Reduce dimensionality w/ PCA before performing clustering (most Genedisco datasets have several hundrerds of input dimensions)
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=n_components)
    x = pca.fit_transform(x)
    #Get optimal number of clusters
    if num_clusters is None:
        optimal_num_clusters = find_optimal_number_clusters(x, min_components=2, max_components=int(len(row_names)/4), num_repeats=10, plot_location=plot_location)
        print("Optimal number of clusters {}".format(optimal_num_clusters))
    else:
        optimal_num_clusters = num_clusters
    #Refit GMM with optimal number of clusters
    GMM_model = GMM(n_components=optimal_num_clusters, covariance_type='full', max_iter=1000, n_init=20, tol=1e-4).fit(x)
    labels = GMM_model.predict(x)
    dict_cluster_id_item_name=defaultdict(list)
    dict_item_name_cluster_id={}
    for index in range(len(row_names)):
        dict_cluster_id_item_name[labels[index]].append(dict_index_to_item[index])
        dict_item_name_cluster_id[dict_index_to_item[index]] = labels[index]
    with open(path_cache_cluster_files_prefix+f'_topk_{optimal_num_clusters}_clusters_to_items.pkl', "wb") as fp:
        pkl.dump(dict_cluster_id_item_name, fp)
    with open(path_cache_cluster_files_prefix+f'_topk_items_to_{optimal_num_clusters}_clusters.pkl', "wb") as fp:
        pkl.dump(dict_item_name_cluster_id, fp)
    return x, dict_cluster_id_item_name, dict_item_name_cluster_id


# Load the data

# === Read from souce ===
# achilles_h5 = HDF5DataSource("achilles.h5", fill_missing_value=0)
# sanchez_2021_neurons_tau_h5 = HDF5DataSource(fn + ".h5", duplicate_merge_strategy=sp.MeanMergeStrategy())

# # check the fields in the data
# dataset_x = sp.CompositeDataSource([achilles_h5]) # 17654
# dataset_y = sp.CompositeDataSource([sanchez_2021_neurons_tau_h5]) # 17983

# dict_index_to_name = dataset_x.reverse_row_index # OrderedDict
# dict_name_to_index = dataset_x.row_index # OrderedDict

# available_indices = sorted(
#     list(
#         set(dataset_x.get_row_names()).intersection(
#             set(dataset_y.get_row_names())
#         )
#     )
# )
# === ===

# dataset_x = dataset_x.subset(available_indices) # pd.DataFrame(dataset_x.get_data()[0]) -> (17176, 808)
# # check length of dataset_x.get_row_names() -> 17176
# dataset_y = dataset_y.subset(available_indices) # pd.DataFrame(dataset_y.get_data()[0]) -> (17176, 1)

# df_x = pd.DataFrame(dataset_x.get_data()[0], index=dataset_x.get_row_names(), columns=dataset_x.get_column_names())
# df_y = pd.DataFrame(dataset_y.get_data()[0], index=dataset_y.get_row_names())
# df_x['y'] = df_y

problem_lst = [
    "schmidt_2021_ifng",
    "schmidt_2021_il2",
    "zhuang_2019",
    "sanchez_2021_tau",
    "zhu_2021_sarscov2_host_factors",
]

# for fn in problem_lst:
#     df = pd.read_csv(fn + ".csv", index_col=0)
#     print(df.shape)

fn = problem_lst[0]
df = pd.read_csv(fn + ".csv", index_col=0)
df_x = df.drop(columns=["y"])
df_y = df["y"]

available_indices = list(df.index)
#%%
# save a small subset of data to testing
def topk_indices(y: pd.Series, k: float):
    '''
    Returns:
        list of strings (indices)
    '''
    return list(y.sort_values(ascending=False).index[:int(k * y.shape[0])])
topk_idx = topk_indices(df_y, 0.1)

topk_df = df.loc[topk_idx]
# topk_indices = get_topk_indices(dataset_y, 0.1)

topk_x = torch.tensor(topk_df.drop(columns=["y"]).values, dtype=torch.float64)

dim = 20

topk_x = StandardScaler().fit_transform(topk_x)
pca = PCA(n_components=dim)
topk_x = pca.fit_transform(topk_x)

topk_df = pd.DataFrame(topk_x, index=topk_df.index)
topk_df['y'] = df_y.loc[topk_df.index]

topk_df.to_csv("test_schmidt_2021_ifng.csv")

#%%



rand_idx = np.random.choice(available_indices, 5)

#%%
def get_data_torch(idx):
    x = torch.tensor(df.loc[idx].values, dtype=torch.float64)
    y = torch.tensor(df.loc[idx], dtype=torch.float64)
    return x, y

x_torch = get_data_torch(rand_idx)
#%%

def subset_select(v, h_sampler, subset_size, budget=20):
    # for moment, just do monte carlo estimate
    # h_sampler : v -> h(v, eta), with eta sampled from some distribution
    # out_fn = either gaussian additive noise or multiplicative bernoulli sampled from GP classifier
    values = np.asarray([h_sampler(v) for _ in range(budget)])
    mx = random_argmax(np.mean(values, axis=0))
    idxes = [mx]
    n = len(v)
    for i in range(subset_size - 1):
        e_vals = np.zeros(n)
        for j in range(len(v)):
            test_idxes = idxes
            if j not in idxes:
                test_idxes = idxes + [j]
                test_idxes = np.asarray(test_idxes)
                e_vals[j] = np.mean(np.max(values[:, test_idxes], axis=-1))
        idxes.append(random_argmax(e_vals))
    return idxes

def random_argmax(vals):
    max_val = np.max(vals)
    idxes = np.where(vals == max_val)[0]
    return np.random.choice(idxes)



x_pca, topk_clusters_to_item, topk_item_to_clusters = get_top_target_clusters(
    dataset_x,
    topk_indices,
    num_clusters=20,
    n_components=20,
    path_cache_cluster_files_prefix="clusters_sanchez_2021_tau_achilles_0.1",
)



# %%

x_np = dataset_x.get_data()[0]
#%%
lst = []
for i in range(len(available_indices)):
    if dataset_x.reverse_row_index[i] != dataset_y.reverse_row_index[i]:
        lst.append(i)
    
# %%
