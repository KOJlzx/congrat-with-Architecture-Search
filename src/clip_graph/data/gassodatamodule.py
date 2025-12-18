'''
The pytorch-lightning LightningDataModules encapsulating our datasets and their
splits, loaders, etc.
'''

from typing import List, Optional, Any, Union

import os
import pickle
import collections as cl
import multiprocessing as mp
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import networkx as nx

import torch
import torch.utils.data as td
import transformers as tf

import sentence_transformers as st

import pytorch_lightning as pl

from sklearn.decomposition import TruncatedSVD

from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_dense_adj, to_undirected

from .gassodataset import (
    GraphDataset, GraphTextDataset,
    BatchGraphTextDataset
)

from .datamodule import BaseDataModule, TextDataModule
from ..utils import _train_val_test_split
import time

#
# Base data module functionality
#


#
# Text-only data module
#




#
# Graph-only data module
#


class GraphDataModule(BaseDataModule):
    def __init__(self,
        drop_isolates: bool = True,
        directed: bool = False,

        **kwargs: Any
    ) -> None:
        if 'batch_size' in kwargs.keys() and kwargs['batch_size'] != 1:
            raise ValueError('batch_size must be 1 for GraphDataModule')
        elif 'batch_size' not in kwargs.keys():
            kwargs['batch_size'] = 1

        super().__init__(**kwargs)

        self.drop_isolates = drop_isolates
        self.directed = directed

        self._graph_data_cache_path = os.path.join(self.data_dir, 'graph_data.pkl')

    @abstractmethod
    def prep_graph(self,
        edgelist: pd.DataFrame,
        node_data: pd.DataFrame
    ) -> Data:
        raise NotImplementedError()

    def _ensure_graph_object(self) -> None:
        if os.path.exists(self._graph_data_cache_path):
            return

        graph_data = self.prep_graph(
            edgelist = self._get_edgelist(),
            node_data = self._get_node_data()
        )

        with open(self._graph_data_cache_path, 'wb') as f:
            pickle.dump(graph_data, f)

    def _get_graph_object(self) -> Data:
        print('in_get_graph_object')
        with open(self._graph_data_cache_path, 'rb') as f:
            ret = pickle.load(f)
        print('in_get_graph_object_pickle_load')
        if not self.directed:
            if hasattr(ret, 'num_nodes'):
                num_nodes = ret.num_nodes
            elif hasattr(ret, 'x'):
                num_nodes = ret.x.shape[0]
            else:
                num_nodes = None

            ret.edge_index = to_undirected(ret.edge_index, num_nodes=num_nodes)

        return ret

    def prepare_data(self) -> "GraphDataModule":
        self._ensure_graph_object()

        return self

    def setup(self, stage: Optional[str] = None) -> "BaseDataModule":
        self.dataset = GraphDataset(
            graph_data=self._get_graph_object(),
            drop_isolates=self.drop_isolates,
            device = self.device,
        )

        self.split()

        return self


#
# Graph+text data module
#


class GraphTextDataModule(TextDataModule, GraphDataModule):
    def __init__(self,
        random_data_debug: bool = False,
        max_texts_per_node: int = 0,
        transductive: bool = False,
        transductive_identity_features: bool = False,

        **kwargs: Any
    ) -> None:
        assert not (not transductive and transductive_identity_features), \
            "transductive_identity_features requires transductive = True"

        batch_size = kwargs.pop('batch_size', 32)
        print('batch_size', batch_size)
        super().__init__(**kwargs)

        self.max_texts_per_node = max_texts_per_node
        self.transductive = transductive
        self.transductive_identity_features = transductive_identity_features

        self.random_data_debug = random_data_debug
        self._real_batch_size = batch_size

    def prepare_data(self) -> "GraphTextDataModule":
        self._ensure_graph_object()

        return self

    def setup(self, node: int = 0, world_size: int = 1, k_hop: int = 1, include_self: bool = True, idx: Optional[int] = None, stage: Optional[str] = None) -> "GraphTextDataModule":
        self.dataset = GraphTextDataset(
            graph_data = self._get_graph_object(),
            drop_isolates = self.drop_isolates,
            text = self._get_text(),
            tokenizer_name = self.tokenizer_name,
            mlm = self.mlm,
            mlm_probability = self.mlm_probability,
            random_data_debug = self.random_data_debug,
            max_texts_per_node = self.max_texts_per_node,
            transductive = self.transductive,
            transductive_identity_features = self.transductive_identity_features,
            device = self.device,
        )

        print("in_setup_graph_text_dataset_extract_subgraph_with_texts")
        total_samples = self.dataset.graph_data.node_ids.shape[0]
        samples_per_shard = total_samples // world_size
        start_idx = node * samples_per_shard
        end_idx = start_idx + samples_per_shard if node < world_size - 1 else total_samples
        node_mask = torch.zeros(total_samples, dtype=torch.bool)
        node_mask[start_idx:end_idx] = True


        self.dataset.extract_subgraph_with_texts(node_mask, k_hop, include_self)
        print("in_setup_graph_text_dataset")
        self.split()
        print("in_setup_graph_text_dataset_split")
        # wrap the datasets in batching logic
        datasets = ['train_dataset', 'val_dataset', 'test_dataset']
        for dataset_name in datasets:
            dataset = getattr(self, dataset_name)
            dataset.compute_mutuals()
            print('in_setup_graph_text_dataset_compute_mutuals')    
            dataset = BatchGraphTextDataset(
                dataset,
                batch_size = self._real_batch_size,
                seed = self.seed
            )

            setattr(self, dataset_name, dataset)

        return self


#
# Data-specific prep code
#


class SVDMixin:
    @property
    def svd_vectors_name(self):
        raise NotImplementedError()

    def post_compute_split_nodes_hook(self) -> "SVDMixin":
        svd_dim = 768
        adj = to_dense_adj(self.dataset.graph_data.edge_index).squeeze(dim=0)
        all_nodes = self.dataset.graph_data.node_ids

        train_nodes = self._split_nodes['train']
        train_mask = torch.isin(all_nodes, train_nodes)
        train_adj = adj[train_mask, :][:, train_mask].numpy()

        # use of train_mask on the columns is not an error! need same
        # number of features as for train because we want to project the
        # validation and test data points into the training data's space
        val_nodes = self._split_nodes['val']
        val_mask = torch.isin(all_nodes, val_nodes)
        val_adj = adj[val_mask, :][:, train_mask].numpy()

        test_nodes = self._split_nodes['test']
        test_mask = torch.isin(all_nodes, test_nodes)
        test_adj = adj[test_mask, :][:, train_mask].numpy()

        # FIXME should we use algorithm='arpack' to make this completely
        # deterministic?
        mod = TruncatedSVD(svd_dim, algorithm='randomized').fit(train_adj)
        train_svd = torch.from_numpy(mod.transform(train_adj))
        val_svd = torch.from_numpy(mod.transform(val_adj))
        test_svd = torch.from_numpy(mod.transform(test_adj))

        svd = torch.cat([train_svd, val_svd, test_svd], dim=0)
        svd_nodes = torch.cat([
            all_nodes[train_mask],
            all_nodes[val_mask],
            all_nodes[test_mask]
        ], dim=0)

        # has to be same order as in the dataset object
        svd_inds = (all_nodes[:, None] == svd_nodes).nonzero()[:, 1]

        setattr(
            self.dataset.graph_data,
            self.svd_vectors_name,
            svd[svd_inds, :]
        )

        return self


class PubmedDataMixin(SVDMixin):
    svd_vectors_name = 'x'

    def prep_graph(self,
        edgelist: pd.DataFrame,
        node_data: pd.DataFrame
    ) -> Data:
        G = nx.from_pandas_edgelist(edgelist, create_using=nx.DiGraph)

        # edgelists don't include isolates
        isolates = node_data.loc[~node_data.node_id.isin(G.nodes), 'node_id']
        for node in isolates.tolist():
            G.add_node(node)

        tmp = node_data[['node_id', 'label']].set_index('node_id')
        tmp['node_ids'] = tmp.index.copy()
        tmp = tmp.to_dict('index')
        nx.set_node_attributes(G, tmp)

        ret = from_networkx(G)

        del ret.num_nodes

        return ret



#
# Put the data mixins and the other classes together
#

class PubmedGraphDataModule(PubmedDataMixin, GraphDataModule):
    pass


class PubmedGraphTextDataModule(PubmedDataMixin, GraphTextDataModule):
    pass

