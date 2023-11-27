import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if True:
    from datasets import load_dataset
    from stats import get_test_only_receivers
    from tgn_attn.module import TGNLinkPredictor
    from shared_utils import train_val_test_split_dataframe
    from neg_sampling import (
        HistoricalDestinationNegativeSampler,
        HistoricalNegativeSampler,
        RandomDestinationNegativeSampler,
    )

    from stats import get_test_only_edges

"""
For each event, calculate the score of the event against a few negative samples 

"""
# NEG_SAMPLING = "historical"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CKPT_PATH = "models/wikipedia/3/checkpoint_10.pth"
CKPT_PATH = "models/enron/4/checkpoint_30.pth"

RUN_PATH = os.path.dirname(CKPT_PATH)

DATASET = os.path.basename(os.path.dirname(RUN_PATH))

SCORES_PATH = os.path.join(
    CKPT_PATH.rstrip(".pth") + f"_scores.csv",
)
print(SCORES_PATH)

events, event_features, node_features = load_dataset(
    DATASET,
    data_dir=os.path.expanduser("~/data/TG_network_datasets"),
).values()

# events = events.iloc[:1000]
num_nodes, num_node_features = node_features.shape
# Split the dataset into train val and test, essentially annotates the events dataframe with a column "split"
train_val_test_split_dataframe(events, val_ratio=0.15, test_ratio=0.15)

# Add 1 negative event for each positive event
# Negative Sampler for training: random destination

# Build torch Data
data = TemporalData(
    src=torch.tensor(events["src"].values),
    dst=torch.tensor(events["dst"].values),
    t=torch.tensor(events["t"].values).long(),
    msg=torch.tensor(events["label"].values).reshape(-1, 1).float(),
)
data.to(DEVICE)
data_loader = TemporalDataLoader(
    data,
    batch_size=200,
)
# Load pretrained model
# fpath = "tmp/tgn_link_pred_state_dict.pt"


train_src = events.loc["train", "src"].values
train_dst = events.loc["train", "dst"].values
dst_observed_during_test_only = get_test_only_receivers(events)
dst_observed_during_training = np.unique(train_dst)
src_test_only, dst_test_only = get_test_only_edges(events)
neg_samplers = {
    "destination": RandomDestinationNegativeSampler(n_nodes=len(node_features)),
    "historical_edge": HistoricalNegativeSampler(train_src, train_dst),
    "inductive_edge": HistoricalNegativeSampler(src_test_only, dst_test_only),
    "inductive_dst": HistoricalDestinationNegativeSampler(
        dst_observed_during_test_only
    ),
    "historical_dst": HistoricalDestinationNegativeSampler(
        dst_observed_during_training
    ),
}


tgn_link_pred = TGNLinkPredictor(
    num_nodes=num_nodes,
    all_t=data.t,
    all_msg=data.msg,
    device=DEVICE,
).to(DEVICE)
statedict = torch.load(CKPT_PATH)
tgn_link_pred.load_state_dict(statedict)
# Disable gradient computation
torch.set_grad_enabled(False)
tgn_link_pred.memory.reset_state()  # Start with a fresh memory.
tgn_link_pred.neighbor_loader.reset_state()  # Start with an empty graph.
# Calculate the scores on all the events against their negative alter ego

tgn_link_pred.eval()

tgn_link_pred.memory.reset_state()  # Start with a fresh memory.
tgn_link_pred.neighbor_loader.reset_state()
metrics = defaultdict(list)
pbar = tqdm(data_loader, desc="Calculating Scores for all events")
scores = defaultdict(list)


for batch in pbar:
    # First calculate the score of the true event
    pos_out = tgn_link_pred(batch.src, batch.dst, batch.t, batch.msg)
    scores["pos"] += pos_out.cpu().ravel().tolist()

    for ns_name, neg_sampler in neg_samplers.items():
        # Then calculate the score of the negative events
        # Provided by the different negative samplers

        neg_src, neg_dst = neg_sampler(batch.src.cpu().numpy(), batch.dst.cpu().numpy())
        neg_dst = torch.tensor(neg_dst.ravel()).to(DEVICE)
        neg_src = torch.tensor(neg_src.ravel()).to(DEVICE)

        neg_out = tgn_link_pred(neg_dst, neg_src, batch.t, batch.msg)

        scores[ns_name] += neg_out.cpu().ravel().tolist()

    # Update the memory and get ready to score the next batch
    tgn_link_pred.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
    tgn_link_pred.neighbor_loader.insert(batch.src, batch.dst)

# Save the scores
scores = pd.DataFrame(scores)

print("Saving scores to ", SCORES_PATH)
scores.to_csv(SCORES_PATH, index=False)
