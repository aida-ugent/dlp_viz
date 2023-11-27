import os
import urllib
import zipfile

import numpy as np
import pandas as pd

dgb_datasets = [
    "CanParl",
    "Contacts",
    "enron",
    "Flights",
    "lastfm",
    "mooc",
    "reddit",
    "SocialEvo",
    "uci",
    "UNtrade",
    "UNvote",
    "USLegis",
    "wikipedia",
]


def download_dgb_data(dataset_name: str, data_dir="./data"):
    if dataset_name not in dgb_datasets:
        raise ValueError(f"{dataset_name} is not a valid dataset name")
        return
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    fpath = os.path.join(data_dir, dataset_name)
    if os.path.exists(fpath):
        print(f"{dataset_name} dataset found, loading it")
        return
    print(f"Downloading {dataset_name} dataset")
    url = f"https://zenodo.org/records/7213796/files/{dataset_name}.zip?download=1"
    urllib.request.urlretrieve(url, fpath)

    print(f"Extracting {dataset_name} dataset")
    with zipfile.ZipFile(fpath, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    print(f"Removing {dataset_name} zip file")
    os.remove(fpath)

    print(f"Done!")


def load_dgb_data(dataset_name, data_dir="./data"):
    download_dgb_data(dataset_name, data_dir)
    data_dir = os.path.join(data_dir, dataset_name)
    ml = pd.read_csv(
        data_dir + f"/ml_{dataset_name}.csv",
        index_col=0,
        header=0,
        names=["src", "dst", "t", "label", "idx"],
    )
    ml_event_feat = np.load(data_dir + f"/ml_{dataset_name}.npy")
    ml_node_feat = np.load(data_dir + f"/ml_{dataset_name}_node.npy")
    return {
        "events": ml,
        "event_features": ml_event_feat,
        "node_features": ml_node_feat,
    }
