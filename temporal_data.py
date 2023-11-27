# from utils.viz import get_animation_df
import os

import numpy as np
import torch
import torch.utils.data as data
import torch_geometric.data as data


class TemporalData(data.TemporalData):
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    def __init__(
        self,
        src,
        dst,
        t,
        num_nodes=None,
        full_edge_index=True,
        *args,
        **kwargs,
    ):
        super().__init__(src, dst, t, *args, **kwargs)
        if num_nodes is not None:
            self.n_nodes = num_nodes
        else:
            self.n_nodes = self.num_nodes

        # Those are the unique edges
        self.pos_edge_index = torch.unique(torch.vstack([self.src, self.dst]), dim=1)

        # Attributes where we will store the binary edge labels for the train/val/test split
        self.edge_label_index = None
        self.edge_label_time = None
        self.statistics = {
            "Number of nodes": self.num_nodes,
            "Number of unique edges": self.num_unique_edges,
            "Number of events": self.num_events,
        }
        # All the datasets need to implement either an attribute or a property
        self.edge_index = None
        self.edge_label = None
        if full_edge_index:
            self.edge_index = self.get_full_edge_index()
            self.edge_label = self.get_edge_label(self.edge_index)
        self.src_nodes = torch.unique(self.src)
        self.num_node_pairs = self.num_nodes * (self.num_nodes - 1)
        self.num_node_combinations = self.num_nodes * (self.num_nodes - 1) // 2
        # If the number of nodes is relatively small, we store all the possible edges, along with an indicator of whether they are present or not at any time

    def get_edge_index_label(self):
        """Eventually override this to do negative sampling"""
        return self.edge_index, self.edge_label

    def __getitem__(self, idx):
        """
        temporal_data[idx] returns all the data where the source node is idx:
        """
        mask = np.in1d(self.src, idx)
        edge_index, edge_label = self.get_edge_index_label()
        mask_edge_index = np.in1d(edge_index[0], idx)
        return (
            # Events src,dst,t
            self.src[mask],
            self.dst[mask],
            self.t[mask],
            # Indices of the possible edges, along with labels
            # Indicating whether they are present or not
            edge_index[:, mask_edge_index],
            edge_label[mask_edge_index],
        )

    def __len__(self):
        return self.num_nodes

    def collate(self, batches):
        src = torch.hstack([b[0] for b in batches])
        dst = torch.hstack([b[1] for b in batches])
        t = torch.hstack([b[2] for b in batches])

        edge_index = torch.hstack([b[3] for b in batches])
        edge_label = torch.hstack([b[4] for b in batches])
        return src.long(), dst.long(), t.float(), edge_index.long(), edge_label.long()
