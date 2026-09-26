import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tests.carrot_router import (
    compute_cost_denominator,
    load_metadata,
    load_soft_oracle_metadata,
    pack_soft_oracle_labels,
    regret_gap_weights,
    soft_oracle_loss,
    softmax_numpy,
    unpack_soft_oracle_labels,
)


class SoftOracleDataTest(unittest.TestCase):
    def test_mean_row_max_denominator(self):
        cost = np.asarray(
            [
                [1.0, 3.0, 2.0],
                [4.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.assertAlmostEqual(
            compute_cost_denominator(cost, "mean-row-max"),
            3.5,
        )
        self.assertAlmostEqual(
            compute_cost_denominator(cost, "global-max"),
            4.0,
        )

    def test_soft_targets_sum_to_one(self):
        reward = np.asarray(
            [[0.2, 0.4, 0.1], [0.8, -0.1, 0.3]],
            dtype=np.float32,
        )
        targets = softmax_numpy(reward, 0.05)
        np.testing.assert_allclose(
            targets.sum(axis=1),
            np.ones(2),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertTrue(np.all(targets >= 0.0))

    def test_regret_weights_are_clipped_top_two_gaps(self):
        reward = np.asarray(
            [
                [0.9, 0.5, 0.1],
                [2.0, 0.2, -1.0],
                [0.4, 0.4, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            regret_gap_weights(reward),
            np.asarray([0.4, 1.0, 0.0], dtype=np.float32),
        )

    def test_pack_round_trip(self):
        targets = np.asarray([[0.7, 0.3]], dtype=np.float32)
        reward = np.asarray([[0.5, 0.2]], dtype=np.float32)
        weights = np.asarray([0.3], dtype=np.float32)
        packed = pack_soft_oracle_labels(targets, reward, weights)
        actual_targets, actual_reward, actual_weights = (
            unpack_soft_oracle_labels(packed, 2)
        )
        np.testing.assert_array_equal(actual_targets, targets)
        np.testing.assert_array_equal(actual_reward, reward)
        np.testing.assert_array_equal(actual_weights, weights)


class SoftOracleLossTest(unittest.TestCase):
    def test_matching_logits_have_lower_loss(self):
        targets = np.asarray(
            [[0.8, 0.15, 0.05], [0.1, 0.2, 0.7]],
            dtype=np.float32,
        )
        reward = np.zeros_like(targets)
        weights = np.ones(2, dtype=np.float32)
        labels = torch.tensor(
            pack_soft_oracle_labels(targets, reward, weights)
        )
        matching_logits = torch.log(torch.tensor(targets))
        reversed_logits = matching_logits.flip(dims=[1])
        matching_loss = soft_oracle_loss(matching_logits, labels, 3)
        reversed_loss = soft_oracle_loss(reversed_logits, labels, 3)
        self.assertLess(matching_loss.item(), reversed_loss.item())

    def test_zero_weight_batch_is_finite_and_zero(self):
        targets = np.asarray([[0.5, 0.5]], dtype=np.float32)
        reward = np.asarray([[0.3, 0.3]], dtype=np.float32)
        weights = np.asarray([0.0], dtype=np.float32)
        labels = torch.tensor(
            pack_soft_oracle_labels(targets, reward, weights)
        )
        loss = soft_oracle_loss(torch.zeros((1, 2)), labels, 2)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.0)


class MetadataCompatibilityTest(unittest.TestCase):
    def test_legacy_metadata_still_loads(self):
        metadata = {
            "profile": "qwen3-8b-l4",
            "model_name": "Qwen/Qwen3-8B-Base",
            "model_names": ["Model_A", "Model_B"],
            "text_column": "query",
            "max_length": 512,
            "cost_mean": [0.1, 0.2],
            "cost_std": [0.01, 0.02],
            "global_max_cost": 1.37602,
            "cost_weight": 0.15,
            "seed": 42,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            path.write_text(json.dumps(metadata), encoding="utf-8")
            loaded = load_metadata(Path(directory))
        self.assertAlmostEqual(loaded.global_max_cost, 1.37602)
        self.assertEqual(loaded.ranking_loss_weight, 1.0)

    def test_soft_oracle_metadata_loads(self):
        metadata = {
            "profile": "qwen3-8b-l4",
            "model_name": "Qwen/Qwen3-8B-Base",
            "model_names": ["Model_A", "Model_B"],
            "text_column": "query",
            "max_length": 512,
            "cost_denominator_method": "mean-row-max",
            "cost_denominator": 0.0772054,
            "soft_label_temperature": 0.05,
            "seed": 42,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "soft_oracle_metadata.json"
            path.write_text(json.dumps(metadata), encoding="utf-8")
            loaded = load_soft_oracle_metadata(Path(directory))
        self.assertEqual(loaded.cost_denominator_method, "mean-row-max")
        self.assertAlmostEqual(loaded.soft_label_temperature, 0.05)


if __name__ == "__main__":
    unittest.main()
