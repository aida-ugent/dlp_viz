import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)
from tqdm import tqdm


class GraphAttentionEmbedding(torch.nn.Module):
    """
    More precisely, this is a TGAT attention layer.
    It takes as input
        - some node embeddings [n_nodes]: embeddings that need to be combined together.
            This will typically be the memory state of the nodes
        - edge_index []: the connectivity of the graph, that uses the same indexing as the node embeddings.
            This tells which items (row of the node embeddings) are connected to which other items
        - last update: the last time the node embeddings were updated
        - some edge-level features (messages): the eventual messages that come from the input data
        - some edge-level features (time): the time difference between the last update of the source node
    """

    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels,
            out_channels // 2,
            heads=2,
            dropout=0.1,
            edge_dim=edge_dim,
        )

    def forward(
        self,
        x,  #  The node memory states [n_node]
        last_update,  # For each node, the last time it was updated [n_node]
        edge_index,  # Who was connected to who [2, n_edges]
        t,  # The time at which the previous connection was made [n_edges]
        msg,  # The message exchanged at the previous connection [n_edges]
    ):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))  # Size: (num_edges, time_enc_dim)
        edge_attr = torch.cat(
            [rel_t_enc, msg], dim=-1
        )  # Size: (num_edges, time_enc_dim + msg_dim)
        return self.conv(x, edge_index, edge_attr)  # Size: (num_nodes, out_channels)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


class TGNLinkPredictor(nn.Module):
    """
    Basically includes all the parts of the code that require some gradients.
    """

    def __init__(self, num_nodes, all_t, all_msg, device="cpu") -> None:
        super().__init__()
        self.neighbor_loader = LastNeighborLoader(
            num_nodes=num_nodes,
            size=10,
            device=device,
        )
        self.assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

        memory_dim = time_dim = embedding_dim = 100

        # Actual TGN memory module.
        # Given a list of node ids, this module will return the corresponding
        # memory states.
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=1,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(1, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        ).to(device)
        # This combines the memory states associated with the
        # nodes in a give batch, and computes the corresponding
        # Embeddings that can be used to perform the prediction
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=1,
            time_enc=self.memory.time_enc,
        ).to(device)
        self.link_pred = LinkPredictor(embedding_dim).to(device)
        self.all_t = all_t
        self.all_msg = all_msg
        self.memorize = True

    def forward(self, src, dst, t=None, msg=None):
        """
        The forward pass scores the events based on the current state of the memeory
        """
        # All the nodes involved in this batch
        n_id = torch.cat([src, dst]).unique()
        # A local temporal network of the nodes involved in this batch

        n_id, edge_index, e_id = self.neighbor_loader(n_id)

        # Calculate some local indices that allow indexing into z
        self.assoc[n_id] = torch.arange(n_id.size(0), device=src.device)
        # Calculate memory states
        z, last_update = self.memory(n_id)
        # Calculate embedding by combining the memory states using the local
        # Connectivity information and the edge-level features
        z = self.gnn(
            z,
            last_update,
            edge_index,
            self.all_t[e_id],  # Size: (num_edges, 1)
            self.all_msg[e_id],  # Size: (num_edges, 1)
        )

        out = self.link_pred(z[self.assoc[src]], z[self.assoc[dst]])

        return out

    def predict_scores(self, temporal_data, batch_size=200):
        """
        Predict the scores of the edges in the temporal data.
        Events are processed in batches.
        The batching here playes two roles:
        - it is necesary to iteratively update the memory, and calculate the score
        - it allows memory-time tradeoff
        """

        loader = TemporalDataLoader(
            temporal_data,
            batch_size=batch_size,
        )

        self.memory.reset_state()
        self.neighbor_loader.reset_state()
        scores = []
        for batch in tqdm(loader):
            src = batch.src
            dst = batch.dst
            t = batch.t
            msg = batch.msg
            scores.append(self(src, dst, t, msg))
            self.memory.update_state(src, dst, t, msg)
            self.neighbor_loader.insert(src, dst)
        return torch.cat(scores)


class EarlyStopping:
    def __init__(self, min_delta=0.0, value=None, mode="max"):
        self.min_delta = min_delta
        self.value = value
        self.mode = mode

    def __call__(self, value):
        # Return True if we should stop training
        if value is None:
            return False
        else:
            if self.mode == "max":
                if value - self.value > self.min_delta:
                    self.value = value
                    return False
                else:
                    return True
            elif self.mode == "min":
                if self.value - value > self.min_delta:
                    self.value = value
                    return False
                else:
                    return True
            else:
                raise ValueError(f"Unknown mode {self.mode}")
