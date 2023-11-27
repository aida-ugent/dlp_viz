from typing import Any

import numpy as np
from tqdm import trange

from pairing import SzudzikPair
from stats import get_test_only_edges, get_test_only_receivers


def get_neg_samplers(events, nodes):
    train_src = events.loc["train", "src"].values
    train_dst = events.loc["train", "dst"].values

    dst_observed_during_test_only = get_test_only_receivers(events)
    dst_observed_during_training = np.unique(train_dst)
    src_test_only, dst_test_only = get_test_only_edges(events)
    neg_samplers = {
        "historical_edge": HistoricalNegativeSampler(train_src, train_dst),
        "destination": RandomDestinationNegativeSampler(n_nodes=len(nodes)),
        "inductive_edge": HistoricalNegativeSampler(src_test_only, dst_test_only),
        "inductive_dst": HistoricalDestinationNegativeSampler(
            dst_observed_during_test_only
        ),
        "historical_dst": HistoricalDestinationNegativeSampler(
            dst_observed_during_training
        ),
    }
    return neg_samplers


class RandomDestinationNegativeSampler:
    def __init__(self, n_nodes) -> None:
        self.n_nodes = n_nodes

    def __call__(self, src, dst):
        neg_dst = np.random.randint(
            0,
            self.n_nodes,
            size=src.shape[0],
        )
        return src, neg_dst


class HistoricalDestinationNegativeSampler:
    def __init__(self, dst, n_trials=20) -> None:
        self.dst_nodes = np.unique(dst)
        self.n_trials = n_trials

    def __call__(self, src: np.ndarray, dst: np.ndarray) -> Any:
        mask = np.zeros_like(src, dtype=bool)
        neg_dst = np.empty_like(src)
        for _ in range(self.n_trials):
            neg_dst[~mask] = np.random.choice(
                self.dst_nodes,
                size=(~mask).sum(),
            )
            mask = neg_dst != dst
            mask = mask & (neg_dst != src)
            if mask.all():
                break
        return src, neg_dst


class RandomNegativeSamplers:
    def __init__(self, n_nodes, n_neg_samples=1) -> None:
        self.li = SzudzikPair
        self.max_s = self.li.encode(n_nodes, n_nodes)
        self.max_iter = 10
        self.n_neg_samples = n_neg_samples
        pass

    def __call__(self, src, dst):
        mask = np.zeros_like(src, dtype=bool)
        pos_code = self.li.encode(src, dst)
        sample_s = np.empty_like(src)
        sample_src = np.empty_like(src)
        sample_dst = np.empty_like(dst)

        for _ in range(self.max_iter):
            sample_s[~mask] = np.random.randint(
                self.max_s,
                size=((~mask).sum()),
            )
            sample_src, sample_dst = self.li.decode(sample_s)
            mask = sample_src != sample_dst
            mask = mask & (sample_src != src)
            mask = mask & (sample_dst != dst)
            if mask.all():
                break

        return sample_src, sample_dst


class HistoricalNegativeSampler:
    def __init__(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        n_neg_samples: int = 1,
    ) -> None:
        self.li = SzudzikPair
        keys = self.li.encode(src, dst)
        self.historical_keys = np.unique(keys)
        self.n_neg_samples = n_neg_samples
        pass

    def __call__(self, src, dst):
        """Sample negative edges among the edges that have been seen in the past
        (i.e. are stored in self.historical_keys)
        """
        neg_keys = np.random.choice(
            self.historical_keys,
            size=(src.shape[0], self.n_neg_samples),
        )
        neg_src, neg_dst = self.li.decode(neg_keys)
        return neg_src, neg_dst


class NotTrainNegativeSampler:
    """
    Sample only edges that have not been seen in the training set
    """

    def __init__(
        self, src, dst, n_nodes, n_neg_samples=1, max_trial=50, verbose=False
    ) -> None:
        self.li = SzudzikPair
        keys = self.li.encode(src, dst)
        self.historical_keys = np.unique(keys)
        self.n_neg_samples = n_neg_samples
        self.n_nodes = n_nodes
        self.max_s = SzudzikPair.encode(n_nodes, n_nodes)
        self.max_trial = max_trial
        self.trials = trange(self.max_trial, disable=not verbose)

    def __call__(self, src, dst):
        mask = np.zeros_like(src, dtype=bool)
        sample_s = np.empty_like(src)
        sample_src = np.empty_like(src)
        sample_dst = np.empty_like(dst)
        for trial in self.trials:
            sample_s[~mask] = np.random.randint(
                self.max_s,
                size=((~mask).sum()),
            )
            sample_src, sample_dst = SzudzikPair.decode(sample_s)

            # Check that the sampled edge is not in the training set
            mask = ~np.in1d(sample_s, self.historical_keys)
            # We should have at least one of src or dst different from the positive sample
            mask = mask & ((sample_src != src) | (sample_dst != dst))
            mask = mask & (sample_src < self.n_nodes)
            mask = mask & (sample_dst < self.n_nodes)
            mask = mask & (sample_src != sample_dst)

            if mask.all():
                # print("found your samples!")
                break
            if trial == self.max_trial - 1:
                print("reached max iter")

        sample_src = sample_src.reshape(-1, self.n_neg_samples)
        sample_dst = sample_dst.reshape(-1, self.n_neg_samples)
        return sample_src, sample_dst


class TestOnlyEdgesNegativeSampler:
    def __init__(self, src, dst, num_nodes):
        self.li = SzudzikPair
        self.keys = self.li.encode(src, dst)

    def __call__(self, src, dst):
        neg_keys = np.random.choice(
            self.keys,
            size=(src.shape[0], 1),
        )
        neg_src, neg_dst = self.li.decode(neg_keys)
        return neg_src, neg_dst


class HistoricalNodeNegativeSampler:
    def __init__(self, nodes, trials=50):
        self.nodes = nodes
        self.trials = range(trials)

    def __call__(self, src, dst):
        """
        Sample negative source and destination nodes
        among the nodes that have been seen in the past
        """
        mask = np.zeros_like(src, dtype=bool)
        src_neg = np.empty_like(src)
        dst_neg = np.empty_like(dst)
        for _ in self.trials:
            src_neg[~mask] = np.random.choice(self.nodes, size=(~mask).sum())
            dst_neg[~mask] = np.random.choice(self.nodes, size=(~mask).sum())
            mask = src_neg != dst_neg
            mask = mask & (src_neg != src)
            mask = mask & (dst_neg != dst)
            if np.all(mask):
                break

        return src_neg, dst_neg
