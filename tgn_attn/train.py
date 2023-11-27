import os
import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn as nn
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if True:
    import shared_utils
    from datasets import load_dataset
    from neg_sampling import (
        HistoricalDestinationNegativeSampler,
        HistoricalNegativeSampler,
        RandomDestinationNegativeSampler,
    )
    from shared_utils import df_to_temporal_data, train_val_test_split_dataframe
    from stats import get_test_only_edges, get_test_only_receivers
    from tgn_attn.module import TGNLinkPredictor


"""
A simple Training Script for the TGN-Attn model

"""
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


torch.manual_seed(12345)
# Configs
configs = {
    "enron": {
        "lr": 0.00001,
        "weight_decay": 0.0001,
        "n_epochs": 50,
    },
    "wikipedia": {
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "n_epochs": 100,
    },
    "uci": {
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "n_epochs": 100,
    },
}

DATASET = "enron"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Configure output directory
SAVE = True

os.makedirs("models", exist_ok=True)
CKPT_DIR = f"models/{DATASET}"
os.makedirs(CKPT_DIR, exist_ok=True)
runs = os.listdir(CKPT_DIR)
run_id = max([int(x) for x in runs] + [-1]) + 1
CKPT_DIR = os.path.join(CKPT_DIR, str(run_id))
os.makedirs(CKPT_DIR, exist_ok=True)
SAVE_EVERY = 10


# Hyperparameters
cfg = configs[DATASET]
LR, WEIGHT_DECAY, N_EPOCHS = (
    cfg["lr"],
    cfg["weight_decay"],
    cfg["n_epochs"],
)


events, event_features, node_features = load_dataset(
    DATASET,
    data_dir=shared_utils.DATA_DIR,
).values()

# events = events.iloc[, :] # For debugging purposes
# Split the dataset into train val and test, essentially annotates the events dataframe with a column "split"
train_val_test_split_dataframe(events, val_ratio=0.15, test_ratio=0.15)

# Negative Sampler for training: Random Destination Node

neg_sampler = RandomDestinationNegativeSampler(n_nodes=len(node_features))

src_neg, dst_neg = neg_sampler(
    src=events["src"].values,
    dst=events["dst"].values,
)

# Build torch Data
data = TemporalData(
    src=torch.tensor(events["src"].values),
    dst=torch.tensor(events["dst"].values),
    t=torch.tensor(events["t"].values).long(),
    msg=torch.tensor(events["label"].values).reshape(-1, 1).float(),
    src_neg=torch.tensor(src_neg),
    dst_neg=torch.tensor(dst_neg),
)
data.to(DEVICE)
data_loader = TemporalDataLoader(
    data,
    batch_size=200,
)

temporal_data = df_to_temporal_data(events).to(DEVICE)
train_data = df_to_temporal_data(events.loc["train"]).to(DEVICE)
val_data = df_to_temporal_data(events.loc["val"]).to(DEVICE)
test_data = df_to_temporal_data(events.loc["test"]).to(DEVICE)

train_loader = TemporalDataLoader(
    train_data,
    batch_size=200,
)
val_loader = TemporalDataLoader(
    val_data,
    batch_size=200,
)
test_loader = TemporalDataLoader(
    test_data,
    batch_size=200,
)

tgn_link_pred = TGNLinkPredictor(
    num_nodes=len(node_features),
    all_t=temporal_data.t.to(DEVICE),
    all_msg=temporal_data.msg.to(DEVICE),
    device=DEVICE,
)

optimizer = torch.optim.Adam(
    tgn_link_pred.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)
criterion = nn.BCEWithLogitsLoss()


def train_epoch():
    tgn_link_pred.train()

    total_loss = 0
    pbar = tqdm(train_loader, desc="Training Epoch")

    tgn_link_pred.memory.reset_state()  # Start with a fresh memory.
    tgn_link_pred.neighbor_loader.reset_state()
    for batch in pbar:
        optimizer.zero_grad()
        neg_dst = torch.randint(
            0,
            len(node_features),
            (batch.src.size(0),),
            device=DEVICE,
        )
        src_posneg = torch.cat([batch.src, batch.src], dim=0)
        dst_posneg = torch.cat([batch.dst, neg_dst], dim=0)
        t_posneg = torch.cat([batch.t, batch.t], dim=0)
        msg_posneg = torch.cat([batch.msg, batch.msg], dim=0)
        scores = tgn_link_pred(src_posneg, dst_posneg, t_posneg, msg_posneg)
        labels = torch.cat(
            [
                torch.ones(batch.src.size(0)),
                torch.zeros(batch.src.size(0)),
            ],
            dim=0,
        ).to(DEVICE)
        loss = criterion(scores, labels.reshape(-1, 1))

        # Update memory and neighbor loader with ground-truth state.

        tgn_link_pred.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        tgn_link_pred.neighbor_loader.insert(batch.src, batch.dst)
        loss.backward()
        optimizer.step()
        tgn_link_pred.memory.detach()

        total_loss += loss * batch.src.size(0)
    total_loss /= len(train_data)
    return total_loss


# Prepare Negative Samplers for validation

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


@torch.no_grad()
def validation_epoch(loader):
    """
    Evaluate the model on the validation set
    """
    tgn_link_pred.eval()

    tgn_link_pred.memory.reset_state()  # Start with a fresh memory.
    tgn_link_pred.neighbor_loader.reset_state()
    metrics = defaultdict(list)
    pbar = tqdm(loader, desc="Validation Epoch")
    for batch in pbar:
        for ns_name, neg_sampler in neg_samplers.items():
            # Calculate two metrics for the given negative sampler
            # auc_vs_{neg_sampler}, ap_vs_{neg_sampler}

            neg_src, neg_dst = neg_sampler(
                batch.src.cpu().numpy(), batch.dst.cpu().numpy()
            )
            neg_dst = torch.tensor(neg_dst.ravel()).to(DEVICE)
            neg_src = torch.tensor(neg_src.ravel()).to(DEVICE)
            src = torch.stack([batch.src, neg_src], dim=0)
            dst = torch.stack([batch.dst, neg_dst], dim=0)
            t = torch.stack([batch.t, batch.t], dim=0)
            msg = torch.stack([batch.msg, batch.msg], dim=0)

            out = tgn_link_pred(src, dst, t, msg)

            y_pred = out.cpu().ravel()
            y_true = np.concatenate(
                [
                    np.ones(batch.src.size(0)),
                    np.zeros(batch.src.size(0)),
                ]
            ).ravel()
            metrics[f"auc_vs_{ns_name}"].append(roc_auc_score(y_true, y_pred))
            metrics[f"ap_vs_{ns_name}"].append(average_precision_score(y_true, y_pred))
        tgn_link_pred.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        tgn_link_pred.neighbor_loader.insert(batch.src, batch.dst)

    # Average the metrics over all the batches
    for k, v in metrics.items():
        metrics[k] = np.mean(v)
    return metrics


def checkpoint(step):
    state_dict = tgn_link_pred.state_dict()

    fpath = os.path.join(CKPT_DIR, f"checkpoint_{step}.pth")

    torch.save(state_dict, fpath)


# When printing a float, only show 4 decimals
torch.set_printoptions(precision=4)
print("Start training")
pbar = range(N_EPOCHS)
logs = {}
for epoch in pbar:
    print(f"Epoch {epoch}")
    tgn_link_pred.train()
    tgn_link_pred.memory.reset_state()  # Start with a fresh memory.
    tgn_link_pred.neighbor_loader.reset_state()

    train_loss = train_epoch()
    train_loss = train_loss.detach().item()

    epoch_logs = validation_epoch(val_loader)
    epoch_logs["train_loss"] = train_loss

    print(logs)
    if epoch > 1:
        progress = train_loss - last_loss
        rel_decrement = abs(progress / last_loss)

        if progress < 0:
            print(f"Loss decreased by a factor of {rel_decrement:.4f}")
        if progress > 0:
            print(f"Loss increased by a factor of {rel_decrement:.4f}")

    last_loss = train_loss
    if SAVE:
        if epoch % SAVE_EVERY == 0:
            checkpoint(epoch)
        logs[epoch] = dict(epoch_logs)
        # Update logs
        with open(os.path.join(CKPT_DIR, "logs.npy"), "wb") as f:
            np.save(f, logs)

# Save the state dict
checkpoint(N_EPOCHS)

test_ap, test_auc = eval(test_loader)
print("Test APs:", dict(test_ap))
print("Test AUC:", dict(test_auc))
