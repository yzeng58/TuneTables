import gzip
import json
from pathlib import Path
from typing import Optional
from utils import seed_all

try:
    import faiss
except ImportError:
    print("faiss is not available; subset maker will not work until it is installed")

import numpy as np
from sklearn.preprocessing import LabelEncoder

import time
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.random_projection import SparseRandomProjection
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader

from tabpfn.utils import normalize_data, to_ranking_low_mem, remove_outliers, NOP, normalize_by_used_features_f

try:
    import umap
except ImportError:
    print("umap is not available; subset maker umap will not work until it is installed")

class TabularDataset(object):
    def __init__(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        cat_idx: list,
        target_type: str,
        num_classes: int,
        num_features: Optional[int] = None,
        num_instances: Optional[int] = None,
        cat_dims: Optional[list] = None,
        split_indeces: Optional[list] = None,
        split_source: Optional[str] = None,
    ) -> None:
        """
        name: name of the dataset
        X: matrix of shape (num_instances x num_features)
        y: array of length (num_instances)
        cat_idx: indices of categorical features
        target_type: {"regression", "classification", "binary"}
        num_classes: 1 for regression 2 for binary, and >2 for classification
        num_features: number of features
        num_instances: number of instances
        split_indeces: specifies dataset splits as a list of dictionaries, with entries "train", "val", and "test".
            each entry specifies the indeces corresponding to the train, validation, and test set.
        """
        assert isinstance(X, np.ndarray), "X must be an instance of np.ndarray"
        assert isinstance(y, np.ndarray), "y must be an instance of np.ndarray"
        assert (
            X.shape[0] == y.shape[0]
        ), "X and y must match along their 0-th dimensions"
        assert len(X.shape) == 2, "X must be 2-dimensional"
        assert len(y.shape) == 1, "y must be 1-dimensional"

        if num_instances is not None:
            assert (
                X.shape[0] == num_instances
            ), f"first dimension of X must be equal to num_instances. X has shape {X.shape}"
            assert y.shape == (
                num_instances,
            ), f"shape of y must be (num_instances, ). y has shape {y.shape} and num_instances={num_instances}"
        else:
            num_instances = X.shape[0]

        if num_features is not None:
            assert (
                X.shape[1] == num_features
            ), f"second dimension of X must be equal to num_features. X has shape {X.shape}"
        else:
            num_features = X.shape[1]

        if len(cat_idx) > 0:
            assert (
                max(cat_idx) <= num_features - 1
            ), f"max index in cat_idx is {max(cat_idx)}, but num_features is {num_features}"
        assert target_type in ["regression", "classification", "binary"]

        if target_type == "regression":
            assert num_classes == 1
        elif target_type == "binary":
            assert num_classes == 1
        elif target_type == "classification":
            assert num_classes > 2

        self.name = name
        self.X = X
        self.y = y
        self.cat_idx = cat_idx
        self.target_type = target_type
        self.num_classes = num_classes
        self.num_features = num_features
        self.cat_dims = cat_dims
        self.num_instances = num_instances
        self.split_indeces = split_indeces
        self.split_source = split_source

        pass

    def target_encode(self):
        # print("target_encode...")
        le = LabelEncoder()
        self.y = le.fit_transform(self.y)

        # Sanity check
        if self.target_type == "classification":
            assert self.num_classes == len(
                le.classes_
            ), "num_classes was set incorrectly."

    def cat_feature_encode(self):
        # print("cat_feature_encode...")
        if not self.cat_dims is None:
            raise RuntimeError(
                "cat_dims is already set. Categorical feature encoding might be running twice."
            )
        self.cat_dims = []

        # Preprocess data
        for i in range(self.num_features):
            if self.cat_idx and i in self.cat_idx:
                le = LabelEncoder()
                self.X[:, i] = le.fit_transform(self.X[:, i])

                # Setting this?
                self.cat_dims.append(len(le.classes_))

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "cat_idx": self.cat_idx,
            "cat_dims": self.cat_dims,
            "target_type": self.target_type,
            "num_classes": self.num_classes,
            "num_features": self.num_features,
            "num_instances": self.num_instances,
            "split_source": self.split_source,
        }

    @classmethod
    def read(cls, p: Path):
        """read a dataset from a folder"""

        # make sure that all required files exist in the directory
        X_path = p.joinpath("X.npy.gz")
        y_path = p.joinpath("y.npy.gz")
        metadata_path = p.joinpath("metadata.json")
        split_indeces_path = p / "split_indeces.npy.gz"

        assert X_path.exists(), f"path to X does not exist: {X_path}"
        assert y_path.exists(), f"path to y does not exist: {y_path}"
        assert (
            metadata_path.exists()
        ), f"path to metadata does not exist: {metadata_path}"
        assert (
            split_indeces_path.exists()
        ), f"path to split indeces does not exist: {split_indeces_path}"

        # read data
        with gzip.GzipFile(X_path, "r") as f:
            X = np.load(f, allow_pickle=True)
        with gzip.GzipFile(y_path, "r") as f:
            y = np.load(f)
        with gzip.GzipFile(split_indeces_path, "rb") as f:
            split_indeces = np.load(f, allow_pickle=True)

        # read metadata
        with open(metadata_path, "r") as f:
            kwargs = json.load(f)

        kwargs["X"], kwargs["y"], kwargs["split_indeces"] = X, y, split_indeces
        return cls(**kwargs)

    def write(self, p: Path, overwrite=False) -> None:
        """write the dataset to a new folder. this folder cannot already exist"""

        if not overwrite:
            assert ~p.exists(), f"the path {p} already exists."

        # create the folder
        p.mkdir(parents=True, exist_ok=overwrite)

        # write data
        with gzip.GzipFile(p.joinpath("X.npy.gz"), "w") as f:
            np.save(f, self.X)
        with gzip.GzipFile(p.joinpath("y.npy.gz"), "w") as f:
            np.save(f, self.y)
        with gzip.GzipFile(p.joinpath("split_indeces.npy.gz"), "wb") as f:
            np.save(f, self.split_indeces)

        # write metadata
        with open(p.joinpath("metadata.json"), "w") as f:
            metadata = self.get_metadata()
            json.dump(self.get_metadata(), f, indent=4)



class CoresetSampler:
    def __init__(self, number_of_set_points, number_of_starting_points, rand_seed):
        self.number_of_set_points = number_of_set_points
        self.number_of_starting_points = number_of_starting_points
        self.rng = np.random.default_rng(rand_seed)

    def _compute_batchwise_differences(self, a, b):
        return np.sum((a[:, None] - b) ** 2, axis=-1)

    def _compute_greedy_coreset_indices(self, features):
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = self.rng.choice(len(features), number_of_starting_points, replace=False).tolist()
        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = np.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = self.number_of_set_points

        for _ in range(num_coreset_samples):
            select_idx = np.argmax(approximate_coreset_anchor_distances)
            coreset_indices.append(select_idx)
            coreset_select_distance = self._compute_batchwise_differences(
                features, features[select_idx : select_idx + 1]
            )
            approximate_coreset_anchor_distances = np.concatenate(
                [approximate_coreset_anchor_distances, coreset_select_distance],
                axis=-1,
            )
            approximate_coreset_anchor_distances = np.min(
                approximate_coreset_anchor_distances, axis=1
            ).reshape(-1, 1)

        return np.array(coreset_indices)

class SubsetMaker(object):
    def __init__(
        self, subset_features, subset_rows, subset_features_method, subset_rows_method, seed = 135798642,give_full_features=False
    ):

        np.random.seed(seed)

        self.subset_features = subset_features
        self.subset_rows = subset_rows
        self.subset_features_method = subset_features_method
        self.subset_rows_method = subset_rows_method
        self.row_selector = None
        self.feature_selector = None
        self.give_full_features = give_full_features

    def random_subset(self, X, y, action=[]):
        if "rows" in action:
            row_indices = np.random.choice(X.shape[0], self.subset_rows, replace=False)
        else:
            row_indices = np.arange(X.shape[0])
        if "features" in action:
            feature_indices = np.random.choice(
                X.shape[1], self.subset_features, replace=False
            )
        else:
            feature_indices = np.arange(X.shape[1])
        return X[row_indices[:, None], feature_indices], y[row_indices]

    def first_subset(self, X, y, action=[]):
        if "rows" in action:
            row_indices = np.arange(self.subset_rows)
        else:
            row_indices = np.arange(X.shape[0])
        if "features" in action:
            feature_indices = np.arange(self.subset_features)
        else:
            feature_indices = np.arange(X.shape[1])
        return X[row_indices[:, None], feature_indices], y[row_indices]

    def mutual_information_subset(self, X, y, action="features", split="train"):
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be 'train', 'val', or 'test'")
        if split == "train":
            # NOTE: we are only fitting on the first split we see to save time here
            if getattr(self, "feature_selector", None) is None:
                print("Fitting mutual information feature selector ...")
                # start the timer
                timer = time.time()
                self.feature_selector = SelectKBest(
                    mutual_info_classif, k=self.subset_features
                )
                X = self.feature_selector.fit_transform(X, y)
                print(
                    f"Done fitting mutual information feature selector in {round(time.time() - timer, 1)} seconds"
                )
            else:
                X = self.feature_selector.transform(X)
            return X, y
        else:
            X = self.feature_selector.transform(X)
            return X, y

    def pca_subset(self, X, y, action='features', split='train'):
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be 'train', 'val', or 'test'")        
        if split == "train":
            self.feature_selector = PCA(n_components=self.subset_features)
            print("Fitting pca selector ...")
            timer = time.time()
            X = self.feature_selector.fit_transform(X)
            print(f"Done fitting pca feature selector in {round(time.time() - timer, 1)} seconds")
        else:
            X = self.feature_selector.transform(X)
        return X, y

    def K_means_sketch(self, X, y, split='train', fit_first_only=False, rand_seed=0, first_only_num = 1000):
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be 'train', 'val', or 'test'")
        if split == "train":
            rng = np.random.default_rng(rand_seed)
            addl_steps = int(rng.choice(10, 1))
            # This function returns the indices of the k samples that are the closest to the k-means centroids
            X = np.ascontiguousarray(X, dtype=np.float32)
            if fit_first_only:
                X = X[:first_only_num]
            #start the timer
            timer = time.time()
            self.kmeans = faiss.Kmeans(X.shape[1], self.subset_rows, niter=15+addl_steps, verbose=False)
            self.kmeans.train(X)
            print(f"Done fitting kmeans in {round(time.time() - timer, 1)} seconds")
            cluster_centers = self.kmeans.centroids
            index = faiss.IndexFlatL2(X.shape[1])
            index.add(cluster_centers)
            _, indices = index.search(cluster_centers, 1)
            indices = indices.reshape(-1)
            return X[indices], y[indices]
        else:
            return X, y

    def coreset_sketch(self, X, y, split='train', rand_seed=0, number_of_starting_points=5):
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be 'train', 'val', or 'test'")
        if split == "train":        
            # This function returns the indices of the k samples that are a greedy coreset
            number_of_set_points = self.subset_rows  # Number of set points for the greedy coreset
            number_of_starting_points = number_of_starting_points  # Number of starting points for the greedy coreset

            sampler = CoresetSampler(number_of_set_points, number_of_starting_points, rand_seed)
            indices = sampler._compute_greedy_coreset_indices(X)
            return X[indices], y[indices]
        else:
            return X, y



    def ica_subset(self, X, y, action='features', split='train'):
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be 'train', 'val', or 'test'")        
        if split == "train":
            self.feature_selector = FastICA(n_components=self.subset_features)
            print("Fitting ica selector ...")
            timer = time.time()
            X = self.feature_selector.fit_transform(X)
            print(f"Done fitting ica feature selector in {round(time.time() - timer, 1)} seconds")
        else:
            X = self.feature_selector.transform(X)
        return X, y


    def kpca_subset(self, X, y, action='features', split='train'):
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be 'train', 'val', or 'test'")        
        if split == "train":
            self.feature_selector = KernelPCA(n_components=self.subset_features, kernel='rbf', gamma=15)
            print("Fitting kpca selector ...")
            timer = time.time()
            X = self.feature_selector.fit_transform(X)
            print(f"Done fitting kpca feature selector in {round(time.time() - timer, 1)} seconds")
        else:
            X = self.feature_selector.transform(X)
        return X, y

    def isomap_subset(self, X, y, action='features', split='train'):
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be 'train', 'val', or 'test'")        
        if split == "train":
            self.feature_selector = Isomap(n_components=self.subset_features)
            print("Fitting isomap selector ...")
            timer = time.time()
            X = self.feature_selector.fit_transform(X)
            print(f"Done fitting isomaps feature selector in {round(time.time() - timer, 1)} seconds")
        else:
            X = self.feature_selector.transform(X)
        return X, y


    def sparse_random_projection_subset(self, X, y, action='features', split='train'):
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be 'train', 'val', or 'test'")        
        if split == "train":
            self.feature_selector = SparseRandomProjection(n_components=self.subset_features)
            print("Fitting sparse rp selector ...")
            timer = time.time()
            X = self.feature_selector.fit_transform(X)
            print(f"Done fitting sparse feature selector in {round(time.time() - timer, 1)} seconds")
        else:
            X = self.feature_selector.transform(X)
        return X, y


    def locally_linear_embedding_subset(self, X, y, action='features', split='train'):
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be 'train', 'val', or 'test'")        
        if split == "train":
            self.feature_selector = LocallyLinearEmbedding(n_components=self.subset_features)
            print("Fitting lle selector ...")
            timer = time.time()
            X = self.feature_selector.fit_transform(X)
            print(f"Done fitting pca feature selector in {round(time.time() - timer, 1)} seconds")
        else:
            X = self.feature_selector.transform(X)
        return X, y


    def umap_subset(self, X, y, action='features', split='train'):
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be 'train', 'val', or 'test'")        
        if split == "train":
            self.feature_selector = umap.UMAP(n_components=self.subset_features)
            print("Fitting umpa selector ...")
            timer = time.time()
            X = self.feature_selector.fit_transform(X)
            print(f"Done fitting pca feature selector in {round(time.time() - timer, 1)} seconds")
        else:
            X = self.feature_selector.transform(X)
        return X, y

    def tsne_subset(self, X, y, action='features', split='train'):
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be 'train', 'val', or 'test'")        
        if split == "train":
            self.feature_selector = TSNE(n_components=self.subset_features)
            print("Fitting tsne selector ...")
            timer = time.time()
            X = self.feature_selector.fit_transform(X)
            print(f"Done fitting tsne feature selector in {round(time.time() - timer, 1)} seconds")
        else:
            X = self.feature_selector.transform(X)
        return X, y
     

    def pca_subset_white(self, X, y, action='features', split='train'):
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be 'train', 'val', or 'test'")        
        if split == "train":
            self.feature_selector = PCA(n_components=self.subset_features, whiten=True)
            print("Fitting pca white selector ...")
            timer = time.time()
            X = self.feature_selector.fit_transform(X)
            print(f"Done fitting pca feature selector in {round(time.time() - timer, 1)} seconds")
        else:
            X = self.feature_selector.transform(X)
        return X, y

    def make_subset(
        self,
        X,
        y,
        split="train",
        seed=0,
    ):
        """
        Make a subset of the data matrix X, with subset_features features and subset_rows rows.
        :param X: data matrix
        :param y: labels
        :param subset_features: number of features to keep
        :param subset_rows: number of rows to keep
        :param subset_features_method: method to use for selecting features
        :param subset_rows_method: method to use for selecting rows
        :return: subset of X, y
        """
        # print('setting numpy seed to', seed)
        # np.random.seed(seed)

        print("self.subset_features_method",self.subset_features_method)
        if self.give_full_features:
                pass
        else:
            if X.shape[1] > self.subset_features > 0:
                print(
                    f"making {self.subset_features}-sized subset of {X.shape[1]} features ..."
                )
                if self.subset_features_method == "random":
                    X, y = self.random_subset(X, y, action=["features"])
                elif self.subset_features_method == "first":
                    X, y = self.first_subset(X, y, action=["features"])
                elif self.subset_features_method == "mutual_information":
                    X, y = self.mutual_information_subset(
                        X, y, action="features", split=split
                    )
                elif self.subset_features_method == "pca":
                    X, y = self.pca_subset(X, y, action='features', split=split)   
                elif self.subset_features_method == "pca_white":
                    X, y = self.pca_subset_white(X, y, action='features', split=split)   
                elif self.subset_features_method == "ica":
                    X, y = self.ica_subset(X, y, action='features', split=split) 
                elif self.subset_features_method == "kpca":
                    X, y = self.kpca_subset(X, y, action='features', split=split)      
                elif self.subset_features_method == "isomap":
                    X, y = self.isomap_subset(X, y, action='features', split=split)      
                elif self.subset_features_method == "sparse_random_projection":
                    X, y = self.sparse_random_projection_subset(X, y, action='features', split=split)         
                elif self.subset_features_method == "locally_linear_embedding":
                    X, y = self.locally_linear_embedding_subset(X, y, action='features', split=split)   
                elif self.subset_features_method == "umap":
                    X, y = self.umap_subset(X, y, action='features', split=split)   
                elif self.subset_features_method == "tsne":
                    X, y = self.tsne_subset(X, y, action='features', split=split)                                      
                else:
                    raise ValueError(
                        f"subset_features_method not recognized: {self.subset_features_method}"
                    )
                
        if X.shape[0] > self.subset_rows > 0:
            print(f"making {self.subset_rows}-sized subset of {X.shape[0]} rows ...")

            if self.subset_rows_method == "random":
                X, y = self.random_subset(X, y, action=["rows"])
            elif self.subset_rows_method == "first":
                X, y = self.first_subset(X, y, action=["rows"])
            elif self.subset_rows_method == "kmeans":
                X, y = self.K_means_sketch(X, y, split=split, fit_first_only=False, rand_seed=0)
            elif self.subset_rows_method == "coreset":
                X, y = self.coreset_sketch(X, y, split=split, rand_seed=0)
            else:
                raise ValueError(
                    f"subset_rows_method not recognized: {self.subset_rows_method}"
                )


        return X, y


import numpy as np
import pandas as pd


def data_split(X, y, nan_mask): # indices
    x_d = {
        'data': X,
        'mask': nan_mask.values
    }

    if x_d['data'].shape != x_d['mask'].shape:
        raise 'Shape of data not same as that of nan mask!'

    y_d = {
        'data': y.reshape(-1, 1)
    }
    return x_d, y_d


def data_prep(X, y):
    temp = pd.DataFrame(X).fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    X, y = data_split(X, y, nan_mask)
    return X, y



