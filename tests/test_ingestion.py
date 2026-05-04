"""Unit tests for ingestion and temporal splitting."""
from __future__ import annotations
import numpy as np
import pytest

from src.ingestion.elliptic_loader import EllipticLoader
from src.ingestion.temporal_splitter import TemporalSplitter


class TestEllipticLoader:
    """Tests for the dataset loader."""

    def test_synthetic_loads_correctly(self, synthetic_data):
        assert len(synthetic_data.features) == 500
        assert len(synthetic_data.classes) == 500
        # 165 features + time_step = 166 columns
        assert synthetic_data.features.shape[1] == 166

    def test_label_normalization(self):
        loader = EllipticLoader(root="nonexistent")
        # All representations should map correctly
        assert loader._normalize_label("1") == 0    # licit in raw Elliptic
        assert loader._normalize_label("2") == 1    # illicit in raw Elliptic
        assert loader._normalize_label("unknown") == -1
        assert loader._normalize_label("licit") == 0
        assert loader._normalize_label("illicit") == 1

    def test_labels_in_valid_range(self, synthetic_data):
        unique = set(synthetic_data.classes["label"].unique())
        assert unique.issubset({-1, 0, 1})

    def test_time_steps_in_valid_range(self, synthetic_data):
        assert synthetic_data.time_steps.min() >= 1
        assert synthetic_data.time_steps.max() <= 49

    def test_labeled_subset(self, synthetic_data):
        labeled = synthetic_data.labeled_subset()
        assert (labeled["label"] != -1).all()


class TestTemporalSplitter:
    """Tests for the temporal splitter — the most critical correctness component."""

    def test_default_ranges(self, synthetic_data):
        splitter = TemporalSplitter()
        split = splitter.split(synthetic_data)
        assert split.train_range == (1, 30)
        assert split.val_range == (31, 40)
        assert split.test_range == (41, 49)

    def test_no_overlap_train_val(self, synthetic_data):
        splitter = TemporalSplitter()
        split = splitter.split(synthetic_data)
        train_set = set(split.train_tx_ids)
        val_set = set(split.val_tx_ids)
        assert len(train_set & val_set) == 0

    def test_no_overlap_val_test(self, synthetic_data):
        splitter = TemporalSplitter()
        split = splitter.split(synthetic_data)
        val_set = set(split.val_tx_ids)
        test_set = set(split.test_tx_ids)
        assert len(val_set & test_set) == 0

    def test_no_overlap_train_test(self, synthetic_data):
        """The MOST important test — no future leakage into past."""
        splitter = TemporalSplitter()
        split = splitter.split(synthetic_data)
        train_set = set(split.train_tx_ids)
        test_set = set(split.test_tx_ids)
        assert len(train_set & test_set) == 0

    def test_overlapping_ranges_rejected(self):
        with pytest.raises(ValueError):
            TemporalSplitter(train_range=(1, 30), val_range=(25, 40), test_range=(41, 49))

    def test_split_summary(self, synthetic_data):
        split = TemporalSplitter().split(synthetic_data)
        s = split.summary()
        assert "n_train" in s and "n_val" in s and "n_test" in s

    def test_features_and_labels_aligned(self, synthetic_data):
        split = TemporalSplitter().split(synthetic_data)
        feats, labels = TemporalSplitter.get_features_and_labels(
            synthetic_data, split.train_tx_ids,
        )
        assert len(feats) == len(labels)
        assert "time_step" not in feats.columns

    def test_train_time_steps_within_range(self, synthetic_data):
        split = TemporalSplitter().split(synthetic_data)
        train_steps = synthetic_data.time_steps.loc[split.train_tx_ids]
        assert train_steps.min() >= 1
        assert train_steps.max() <= 30

    def test_test_time_steps_within_range(self, synthetic_data):
        split = TemporalSplitter().split(synthetic_data)
        test_steps = synthetic_data.time_steps.loc[split.test_tx_ids]
        assert test_steps.min() >= 41
        assert test_steps.max() <= 49

    def test_labeled_only_filter(self, synthetic_data):
        """When labeled_only=True, no -1 labels should leak into splits."""
        split = TemporalSplitter(labeled_only=True).split(synthetic_data)
        labels_map = synthetic_data.classes.set_index("txId")["label"].to_dict()
        for tx in split.train_tx_ids:
            assert labels_map[tx] != -1
