"""Q4A checks for the GRPO baseline.

These run on CPU with small synthetic tensors; nothing here loads Qwen. The
equivalence and masking assertions are the load-bearing ones -- if padding leaks
into the loss, or if the baseline stops being the reliability-one case of the
general loss, every downstream comparison is invalid.
"""

from __future__ import annotations

import unittest

import torch

from algorithms.qwen_grpo import (
    LossConfig,
    ReliabilityConfig,
    completion_mask,
    exact_match_reward,
    extract_answer,
    extract_gold,
    group_relative_advantage,
    mean_one,
    rwp_grpo_loss,
    token_reliability,
    uniform_grpo_loss,
)


class RewardTest(unittest.TestCase):
    def test_extracts_last_number_and_strips_formatting(self) -> None:
        self.assertEqual(extract_answer("we get 12 then 3,450 apples"), "3450")
        self.assertEqual(extract_answer("costs $18.50 total"), "18.50")
        self.assertIsNone(extract_answer("no digits here"))

    def test_gold_parses_after_the_marker(self) -> None:
        self.assertEqual(extract_gold("blah blah\n#### 72"), "72")
        self.assertIsNone(extract_gold("no marker"))

    def test_reward_is_exact_match(self) -> None:
        self.assertEqual(exact_match_reward("so the answer is 72", "x\n#### 72"), 1.0)
        self.assertEqual(exact_match_reward("so the answer is 71", "x\n#### 72"), 0.0)
        self.assertEqual(exact_match_reward("no number", "x\n#### 72"), 0.0)

    def test_unparseable_completion_scores_zero_not_crash(self) -> None:
        self.assertEqual(exact_match_reward("", "x\n#### 5"), 0.0)


class AdvantageTest(unittest.TestCase):
    def test_standardizes_within_group(self) -> None:
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
        out = group_relative_advantage(rewards, group_size=2)
        self.assertAlmostEqual(float(out[0] + out[1]), 0.0, places=5)
        self.assertGreater(float(out[0]), 0.0)

    def test_degenerate_group_gives_exactly_zero(self) -> None:
        """All-equal rewards carry no signal and must not be amplified."""

        out = group_relative_advantage(torch.tensor([1.0, 1.0, 1.0, 1.0]), 4)
        torch.testing.assert_close(out, torch.zeros(4))

    def test_groups_are_independent(self) -> None:
        rewards = torch.tensor([1.0, 0.0, 5.0, 5.0])
        out = group_relative_advantage(rewards, group_size=2)
        torch.testing.assert_close(out[2:], torch.zeros(2))

    def test_rejects_misaligned_length(self) -> None:
        with self.assertRaises(ValueError):
            group_relative_advantage(torch.zeros(5), group_size=2)


class MaskTest(unittest.TestCase):
    PAD, EOS = 0, 9

    def test_prompt_and_padding_are_excluded(self) -> None:
        seq = torch.tensor([[5, 5, 1, 2, 9, 0, 0]])
        mask = completion_mask(seq, prompt_length=2, pad_token_id=self.PAD,
                               eos_token_id=self.EOS)
        self.assertFalse(bool(mask[0, 0]) or bool(mask[0, 1]))   # prompt
        self.assertFalse(bool(mask[0, 5]) or bool(mask[0, 6]))   # padding

    def test_first_eos_is_kept_and_the_tail_dropped(self) -> None:
        seq = torch.tensor([[5, 1, 9, 3, 9]])
        mask = completion_mask(seq, 1, self.PAD, self.EOS)
        self.assertTrue(bool(mask[0, 1]))
        self.assertTrue(bool(mask[0, 2]))    # the EOS itself is a real decision
        self.assertFalse(bool(mask[0, 3]))
        self.assertFalse(bool(mask[0, 4]))

    def test_sequence_without_eos_keeps_all_generated_tokens(self) -> None:
        seq = torch.tensor([[5, 1, 2, 3]])
        mask = completion_mask(seq, 1, self.PAD, self.EOS)
        self.assertEqual(int(mask.sum()), 3)


class MeanOneTest(unittest.TestCase):
    def test_normalizes_over_active_tokens_only(self) -> None:
        weights = torch.tensor([[1.0, 3.0, 100.0]])
        mask = torch.tensor([[True, True, False]])
        out = mean_one(weights, mask)
        self.assertAlmostEqual(float((out * mask).sum() / mask.sum()), 1.0, places=6)

    def test_empty_mask_is_a_no_op(self) -> None:
        weights = torch.tensor([[1.0, 2.0]])
        mask = torch.zeros_like(weights, dtype=torch.bool)
        torch.testing.assert_close(mean_one(weights, mask), weights)


class LossTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.batch, self.length, self.vocab = 4, 6, 11
        self.logits = torch.randn(self.batch, self.length, self.vocab,
                                  requires_grad=True)
        self.sequences = torch.randint(1, self.vocab, (self.batch, self.length))
        self.mask = torch.zeros(self.batch, self.length, dtype=torch.bool)
        self.mask[:, 2:] = True
        self.advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
        self.config = LossConfig(eta=1.0, lam=1.0, chunk_tokens=3)

    def test_baseline_is_exactly_the_reliability_one_case(self) -> None:
        """D5. If this drifts, the two branches are no longer comparable."""

        ones = torch.ones_like(self.mask, dtype=torch.float32)
        general, _ = rwp_grpo_loss(self.logits, self.sequences, self.mask,
                                   self.advantages, ones, self.config)
        baseline, _ = uniform_grpo_loss(self.logits, self.sequences, self.mask,
                                        self.advantages, self.config)
        torch.testing.assert_close(general, baseline, rtol=0, atol=0)

    def test_padding_cannot_influence_the_loss(self) -> None:
        """Perturbing logits at masked positions must not move the loss."""

        loss_a, _ = uniform_grpo_loss(self.logits, self.sequences, self.mask,
                                      self.advantages, self.config)
        perturbed = self.logits.detach().clone()
        perturbed[:, :2] += 50.0          # masked-out prompt region
        perturbed.requires_grad_(True)
        loss_b, _ = uniform_grpo_loss(perturbed, self.sequences, self.mask,
                                      self.advantages, self.config)
        torch.testing.assert_close(loss_a, loss_b, rtol=1e-6, atol=1e-6)

    def test_chunking_does_not_change_the_loss(self) -> None:
        wide = LossConfig(eta=1.0, lam=1.0, chunk_tokens=1000)
        narrow = LossConfig(eta=1.0, lam=1.0, chunk_tokens=2)
        a, _ = uniform_grpo_loss(self.logits, self.sequences, self.mask,
                                 self.advantages, wide)
        b, _ = uniform_grpo_loss(self.logits, self.sequences, self.mask,
                                 self.advantages, narrow)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_loss_is_finite_and_differentiable(self) -> None:
        loss, stats = uniform_grpo_loss(self.logits, self.sequences, self.mask,
                                        self.advantages, self.config)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(self.logits.grad).all())
        self.assertGreater(stats["active_tokens"], 0)

    def test_zero_advantage_still_produces_a_defined_loss(self) -> None:
        loss, _ = uniform_grpo_loss(self.logits, self.sequences, self.mask,
                                    torch.zeros(self.batch), self.config)
        self.assertTrue(torch.isfinite(loss))

    def test_gradient_accumulation_matches_one_large_batch(self) -> None:
        """Required by Q4A: accumulation must be numerically equivalent."""

        full = self.logits.detach().clone().requires_grad_(True)
        loss, _ = uniform_grpo_loss(full, self.sequences, self.mask,
                                    self.advantages, self.config)
        loss.backward()
        reference = full.grad.clone()

        split = self.logits.detach().clone().requires_grad_(True)
        halves = [slice(0, 2), slice(2, 4)]
        active_total = self.mask[:, 1:].sum()
        for part in halves:
            sub_loss, _ = uniform_grpo_loss(
                split[part], self.sequences[part], self.mask[part],
                self.advantages[part], self.config,
            )
            weight = self.mask[part][:, 1:].sum() / active_total
            (sub_loss * weight).backward()
        torch.testing.assert_close(split.grad, reference, rtol=2e-4, atol=2e-5)


class ReliabilityTest(unittest.TestCase):
    def test_disabled_gives_exactly_one(self) -> None:
        mask = torch.ones(4, 6, dtype=torch.bool)
        out = token_reliability(torch.zeros(4), torch.zeros(4, 6, dtype=torch.long),
                                mask, 2, 2, ReliabilityConfig())
        torch.testing.assert_close(out, torch.ones(4, 6))

    def test_enabled_stays_within_floor_and_one(self) -> None:
        mask = torch.ones(4, 6, dtype=torch.bool)
        config = ReliabilityConfig(enabled=True, floor=0.1)
        tokens = torch.tensor([[1, 1, 2, 2, 3, 3], [1, 1, 5, 5, 3, 3],
                               [1, 1, 2, 2, 3, 3], [1, 1, 7, 7, 3, 3]])
        out = token_reliability(torch.tensor([1.0, 0.0, 1.0, 0.0]), tokens, mask,
                                2, 2, config)
        generated = out[:, 2:]
        self.assertGreaterEqual(float(generated.min()), config.floor - 1e-6)
        self.assertLessEqual(float(generated.max()), 1.0 + 1e-6)

    def test_prompt_region_is_left_at_one(self) -> None:
        mask = torch.ones(2, 5, dtype=torch.bool)
        tokens = torch.tensor([[1, 1, 1, 2, 3], [1, 1, 1, 9, 3]])
        out = token_reliability(torch.tensor([1.0, 0.0]), tokens, mask, 2, 3,
                                ReliabilityConfig(enabled=True))
        torch.testing.assert_close(out[:, :3], torch.ones(2, 3))


if __name__ == "__main__":
    unittest.main()
