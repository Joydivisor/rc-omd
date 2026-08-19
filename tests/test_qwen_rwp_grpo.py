"""Q4B checks: the RWP branch and the Uniform/RWP equivalence suite.

The equivalence assertions are the ones that matter. If reliability-one RWP ever
stops reproducing the baseline bitwise, the two branches are no longer a
controlled comparison and every downstream number is uninterpretable.
"""

from __future__ import annotations

import unittest

import torch

from algorithms.qwen_grpo import (
    LossConfig,
    ReliabilityConfig,
    completion_mask,
    group_relative_advantage,
    rwp_grpo_loss,
    token_reliability,
    uniform_grpo_loss,
)
from algorithms.qwen_rwp_grpo import (
    alignment_diagnostics,
    equivalence_report,
    first_divergence,
    make_rwp_reliability,
)


class BranchConfigTest(unittest.TestCase):
    def test_rwp_reliability_is_always_enabled(self) -> None:
        self.assertTrue(make_rwp_reliability().enabled)
        self.assertTrue(make_rwp_reliability(enabled=False).enabled)

    def test_overrides_are_respected(self) -> None:
        config = make_rwp_reliability(floor=0.25, warmup=4.0)
        self.assertEqual(config.floor, 0.25)
        self.assertEqual(config.warmup, 4.0)


class EquivalenceTest(unittest.TestCase):
    """D5: the baseline is exactly the reliability-one case."""

    def setUp(self) -> None:
        torch.manual_seed(3)
        self.batch, self.length, self.vocab = 6, 8, 13
        self.logits = torch.randn(self.batch, self.length, self.vocab)
        self.sequences = torch.randint(1, self.vocab, (self.batch, self.length))
        self.mask = torch.zeros(self.batch, self.length, dtype=torch.bool)
        self.mask[:, 3:] = True
        self.advantages = group_relative_advantage(
            torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0]), group_size=3
        )

    def test_identical_across_eta_and_lambda(self) -> None:
        for eta in (0.5, 1.0, 2.0):
            for lam in (0.5, 1.0, 3.0):
                with self.subTest(eta=eta, lam=lam):
                    report = equivalence_report(
                        self.logits, self.sequences, self.mask, self.advantages,
                        LossConfig(eta=eta, lam=lam, chunk_tokens=4),
                    )
                    self.assertTrue(report["identical"], report)
                    self.assertEqual(report["absolute_difference"], 0.0)

    def test_lambda_is_irrelevant_when_reliability_is_one(self) -> None:
        """w = r + lam(1-r) collapses to 1, so lambda must cancel entirely."""

        ones = torch.ones_like(self.mask, dtype=torch.float32)
        a, _ = rwp_grpo_loss(self.logits, self.sequences, self.mask,
                             self.advantages, ones,
                             LossConfig(eta=1.0, lam=0.1, chunk_tokens=4))
        b, _ = rwp_grpo_loss(self.logits, self.sequences, self.mask,
                             self.advantages, ones,
                             LossConfig(eta=1.0, lam=9.0, chunk_tokens=4))
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_nonuniform_reliability_actually_changes_the_loss(self) -> None:
        """Otherwise the branch would be a no-op dressed up as a treatment."""

        reliability = torch.full_like(self.mask, 0.3, dtype=torch.float32)
        config = LossConfig(eta=1.0, lam=3.0, chunk_tokens=4)
        weighted, _ = rwp_grpo_loss(self.logits, self.sequences, self.mask,
                                    self.advantages, reliability, config)
        baseline, _ = uniform_grpo_loss(self.logits, self.sequences, self.mask,
                                        self.advantages, config)
        self.assertGreater(float((weighted - baseline).abs()), 1e-6)

    def test_constant_reliability_cancels_in_the_weight_but_not_the_mixture(self) -> None:
        """Reliability acts twice, and only one of the two is normalized away.

        D3's mean-one normalization removes the overall scale of
        ``w = r + lam(1 - r)``, so a constant reliability always yields
        ``w_tilde = 1``. It does NOT remove ``r`` from the mixture
        ``m = [r q + lam(1 - r) pi_old] / w``, whose composition still shifts.
        Asserting full cancellation here would have been wrong, and would have
        hidden the fact that the branch has a real effect even under constant
        reliability.
        """

        config = LossConfig(eta=1.0, lam=1.0, chunk_tokens=4)
        results = []
        for value in (0.5, 0.9):
            reliability = torch.full_like(self.mask, value, dtype=torch.float32)
            loss, stats = rwp_grpo_loss(self.logits, self.sequences, self.mask,
                                        self.advantages, reliability, config)
            results.append((loss, stats))
            # the weight is normalized to exactly one regardless of the constant
            self.assertAlmostEqual(stats["mean_weight"], 1.0, places=5)
        self.assertGreater(float((results[0][0] - results[1][0]).abs()), 1e-4)


class DivergenceTest(unittest.TestCase):
    def test_identical_rollouts_never_diverge(self) -> None:
        sequences = torch.tensor([[7, 1, 2, 3], [7, 1, 2, 3]])
        self.assertEqual(first_divergence(sequences, 1, 2), [3])

    def test_divergence_index_is_relative_to_generation(self) -> None:
        sequences = torch.tensor([[7, 1, 2, 3], [7, 1, 9, 3]])
        self.assertEqual(first_divergence(sequences, 1, 2), [1])

    def test_groups_are_reported_separately(self) -> None:
        sequences = torch.tensor([
            [7, 1, 2], [7, 1, 2],       # group 0, identical
            [7, 5, 2], [7, 6, 2],       # group 1, diverges immediately
        ])
        self.assertEqual(first_divergence(sequences, 1, 2), [2, 0])


class DiagnosticsTest(unittest.TestCase):
    PAD, EOS = 0, 9

    def setUp(self) -> None:
        # two groups of two; the second rollout of each group stops early
        self.sequences = torch.tensor([
            [7, 1, 2, 3, 4],
            [7, 1, 9, 0, 0],
            [7, 5, 6, 7, 8],
            [7, 5, 9, 0, 0],
        ])
        self.mask = completion_mask(self.sequences, 1, self.PAD, self.EOS)
        self.reliability = token_reliability(
            torch.tensor([1.0, -1.0, 1.0, -1.0]), self.mask, 2, 1,
            ReliabilityConfig(enabled=True),
        )

    def test_reports_every_required_metric(self) -> None:
        out = alignment_diagnostics(self.sequences, 1, self.mask, 2,
                                    self.reliability)
        for key in ("first_divergence_mean", "aligned_token_fraction_full_group",
                    "coverage_per_position_mean", "reliability_mean",
                    "reliability_std", "reliability_min", "reliability_max",
                    "reliability_position_trend", "completion_length_mean"):
            self.assertIn(key, out)

    def test_alignment_fraction_falls_when_completions_differ_in_length(self) -> None:
        """The honest measure of how much index alignment the data supports."""

        out = alignment_diagnostics(self.sequences, 1, self.mask, 2,
                                    self.reliability)
        self.assertLess(out["aligned_token_fraction_full_group"], 1.0)
        self.assertGreater(out["aligned_token_fraction_any"],
                           out["aligned_token_fraction_full_group"])

    def test_equal_length_rollouts_align_fully(self) -> None:
        sequences = torch.tensor([[7, 1, 2], [7, 3, 4]])
        mask = completion_mask(sequences, 1, self.PAD, self.EOS)
        reliability = torch.ones_like(mask, dtype=torch.float32)
        out = alignment_diagnostics(sequences, 1, mask, 2, reliability)
        self.assertAlmostEqual(out["aligned_token_fraction_full_group"], 1.0,
                               places=6)

    def test_coverage_is_non_increasing_with_position(self) -> None:
        out = alignment_diagnostics(self.sequences, 1, self.mask, 2,
                                    self.reliability)
        coverage = out["coverage_per_position_mean"]
        for earlier, later in zip(coverage, coverage[1:]):
            self.assertLessEqual(later, earlier + 1e-9)

    def test_reliability_bounds_are_respected(self) -> None:
        out = alignment_diagnostics(self.sequences, 1, self.mask, 2,
                                    self.reliability)
        self.assertGreaterEqual(out["reliability_min"], 0.0)
        self.assertLessEqual(out["reliability_max"], 1.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
