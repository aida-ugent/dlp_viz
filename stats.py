import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from pairing import SzudzikPair


def collect_node_statistics(df):
    """An in-place function that adds the following columns to the dataframe:
    - src_arrival_rank: the arrival rank of the source node
    - dst_arrival_rank: the arrival rank of the destination node
    - src_degree: the degree of the source node
    - dst_degree: the degree of the destination node
    Args:
        df (pd.DataFrame): a dataframe of events
    """
    df.reset_index(inplace=True)
    df["index"] = df.index

    df_dup = pd.concat([df, df]).sort_values("index")
    df_dup["node"] = np.transpose(np.vstack([df.src.values, df.dst.values])).ravel()
    df_dup["node_degree"] = df_dup.groupby("node")["t"].rank(method="first")
    df_dup["role"] = np.tile(["src", "dst"], len(df))
    df_node = (
        df_dup.groupby("node")
        .agg(
            t_min=("t", "min"),
            t_max=("t", "max"),
        )
        .sort_values(["t_min", "t_max"], ascending=True)
    )

    df_node["rank"] = np.arange(len(df_node))

    df["src_arrival_rank"] = df.src.map(df_node["rank"])
    df["dst_arrival_rank"] = df.dst.map(df_node["rank"])

    df_dup = df_dup.set_index("role")

    df["src_degree"] = df_dup.loc["src", "node_degree"].values
    df["dst_degree"] = df_dup.loc["dst", "node_degree"].values


def collect_edge_statistics(df):
    df["edge_key"] = SzudzikPair.encode(df.src, df.dst)

    df["edge_degree"] = df.groupby("edge_key")["t"].rank(method="first").values

    df_edge = df.groupby("edge_key").agg(
        t_min=("t", "min"),
        t_max=("t", "max"),
        edge_count=("t", "count"),
        src=("src", "first"),
        dst=("dst", "first"),
    )
    df_edge.sort_values(["t_min", "t_max"], inplace=True)
    df_edge["rank"] = np.arange(len(df_edge))
    df["edge_arrival_rank"] = df_edge.loc[df.edge_key, "rank"].values


def get_test_only_edges(events):
    """Calculate test only edges"""
    test_mask = events.index.get_level_values("split") == "test"
    train_mask = events.index.get_level_values("split") == "train"
    val_mask = events.index.get_level_values("split") == "val"
    edge_keys = SzudzikPair.encode(events.src, events.dst)
    keys_observed_during_test_only = (
        set(edge_keys.loc[test_mask])
        - set(edge_keys.loc[train_mask])
        - set(edge_keys.loc[val_mask])
    )
    src_test_only, dst_test_only = SzudzikPair.decode(
        np.array(list(keys_observed_during_test_only))
    )

    return src_test_only, dst_test_only


def get_test_only_nodes(events):
    """Calculate test only nodes"""

    node_cols = ["src", "dst"]
    nodes_observed_during_test_only = (
        set(events.loc["test", node_cols])
        - set(events.loc["train", node_cols])
        - set(events.loc["val", node_cols])
    )

    return np.array(list(nodes_observed_during_test_only))


def get_test_only_senders(events):
    """Calculate test only nodes"""

    src_observed_during_test_only = (
        set(events.loc["test", "src"])
        - set(events.loc["train", "src"])
        - set(events.loc["val", "src"])
    )

    return np.array(list(src_observed_during_test_only))


def get_test_only_receivers(events):
    """Calculate test only nodes"""

    dst_observed_during_test_only = (
        set(events.loc["test", "dst"])
        - set(events.loc["train", "dst"])
        - set(events.loc["val", "dst"])
    )

    return np.array(list(dst_observed_during_test_only))


def get_temporal_edge_degree(src, dst, t):
    """
    For each event, return the number of past interactions of the edge just after the event.
    """
    df = pd.DataFrame({"src": src, "dst": dst, "t": t})

    edge_key = ["src", "dst"]

    return df.groupby(edge_key)["t"].rank(method="first").values


def get_temporal_node_degrees(src, dst, t):
    """
    For each event, return the number of past interactions of the source and the destination
    just after the event.
    """
    df = pd.DataFrame({"src": src, "dst": dst, "t": t})

    df_node = pd.concat([df.reset_index(), df.reset_index()]).sort_values("t")
    df_node["node"] = np.transpose(np.vstack([df.src.values, df.dst.values])).ravel()

    df_node["node_event_rank"] = df_node.groupby("node")["t"].rank(method="first")
    df_node["role"] = np.tile(["src", "dst"], len(df))
    df_node = df_node.set_index(["role"])

    src_degree = df_node.loc["src", "node_event_rank"].values
    dst_degree = df_node.loc["dst", "node_event_rank"].values

    return src_degree, dst_degree


def plot_tet(
    df,
    x="x_tet",
    y="t_scaled",
    size="size",
    ax=None,
    *args,
    **kwargs,
):
    ax = plt.gca() if ax is None else ax

    sns.scatterplot(
        data=df,
        x="x_tet",
        y="t_scaled",
        hue="best_ns",
        size="size",
        sizes=(1, 50),
        palette=palette,
        alpha=0.5,
    )
