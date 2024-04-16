"""
Copyright 2022 Pascal Notin, University of Oxford
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import defaultdict

def find_optimal_number_clusters(dataset_x, min_components=2, max_components=100, num_repeats=10, plot_location="temp.png"):
    n_components = np.arange(min_components, max_components)
    models = [GMM(n, covariance_type='full', random_state=0, max_iter=200).fit(dataset_x) for n in n_components for repeat in range(num_repeats)]
    bic = np.array([m.bic(dataset_x) for m in models]).reshape(len(n_components),num_repeats).mean(axis=1).flatten()
    aic = np.array([m.aic(dataset_x) for m in models]).reshape(len(n_components),num_repeats).mean(axis=1).flatten()
    plt.plot(n_components, bic, label='BIC')
    plt.plot(n_components, aic, label='AIC')
    argmin_AIC = min_components + np.argmin(np.array(aic))
    plt.axvline(x = argmin_AIC, color = 'r', label = 'Opt. # clus: {}'.format(argmin_AIC), linestyle='dashed')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.savefig(plot_location)
    plt.title('AIC & BIC Vs number of GMM components')
    return argmin_AIC

def get_top_target_clusters(dataset_x, topk_indices, num_clusters=None, plot_location="temp.png",path_cache_cluster_files_prefix="cluster_file"):
    dataset_x = dataset_x.subset(topk_indices)
    row_names = dataset_x.get_row_names()
    x = dataset_x.get_data()[0]
    dict_index_to_item = {}
    for index,item in enumerate(row_names):
        dict_index_to_item[index] = item
    #Reduce dimensionality w/ PCA before performing clustering (most Genedisco datasets have several hundrerds of input dimensions)
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=20)
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
    return dict_cluster_id_item_name, dict_item_name_cluster_id

if __name__ == "__main__":
    x = np.random.rand(100,10)
    optimal_number_clusters = find_optimal_number_clusters(x,2,80,2)