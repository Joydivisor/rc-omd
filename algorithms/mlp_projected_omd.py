"""Non-linear policy parameterization for `stage-b-mlp-2026-08-19`.

The linear projected algorithms parameterize logits as ``features @ weights``.
These variants replace that with a shared single-hidden-layer tanh MLP applied
to the same feature rows, so the update geometry changes while the task, the
feature structure, and every protocol threshold stay fixed.

Only the parameterization changes. The tabular OMD target, the reliability
estimator, the mixture construction, and the KL accounting are inherited
unchanged from the linear classes, which is what makes the comparison a test of
non-linearity rather than of a different algorithm.
"""

from __future__ import annotations

import numpy as np

from .geometry_omd import ReliabilityWeightedProjectionOMD
from .group_omd import FloatArray, _softmax
from .projected_omd import ProjectedGroupOMD


class MLPHead:
    """Shared ``logits_k = W2 @ tanh(W1 @ f_k + b1) + b2``.

    ``W2`` and both biases start at zero, so the initial policy is exactly
    uniform and matches the linear model's zero-weight initialization. See the
    initialization amendment in `docs/STAGE_B_MLP_PROTOCOL.md`.
    """

    def __init__(
        self, feature_dim: int, n_actions: int, hidden: int, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        self.hidden = int(hidden)
        self.n_actions = int(n_actions)
        self.feature_dim = int(feature_dim)
        self.W1 = rng.normal(
            0.0, 1.0 / np.sqrt(feature_dim), size=(feature_dim, hidden)
        )
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.W2 = np.zeros((hidden, n_actions), dtype=np.float64)
        self.b2 = np.zeros(n_actions, dtype=np.float64)

    @property
    def n_parameters(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def hidden_activations(self, features: FloatArray) -> FloatArray:
        return np.tanh(features @ self.W1 + self.b1)

    def logits(self, features: FloatArray) -> FloatArray:
        return self.hidden_activations(features) @ self.W2 + self.b2

    def parameter_gradient(
        self, features: FloatArray, dlogits: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Backpropagate ``dlogits`` (shape ``(H, A)``) to parameter gradients."""

        hidden = self.hidden_activations(features)
        gW2 = hidden.T @ dlogits
        gb2 = dlogits.sum(axis=0)
        dhidden = dlogits @ self.W2.T
        dpre = dhidden * (1.0 - hidden * hidden)
        gW1 = features.T @ dpre
        gb1 = dpre.sum(axis=0)
        return gW1, gb1, gW2, gb2

    def step(self, gradients, learning_rate: float) -> None:
        gW1, gb1, gW2, gb2 = gradients
        self.W1 -= learning_rate * gW1
        self.b1 -= learning_rate * gb1
        self.W2 -= learning_rate * gW2
        self.b2 -= learning_rate * gb2

    def flat_parameters(self) -> FloatArray:
        return np.concatenate(
            [self.W1.ravel(), self.b1.ravel(), self.W2.ravel(), self.b2.ravel()]
        )

    def position_jacobians(self, features: FloatArray) -> FloatArray:
        """Per-position Jacobian of the logit vector w.r.t. every parameter.

        Returns shape ``(H, A * n_parameters)``: row ``k`` is the flattened
        Jacobian ``d logits_k / d theta``. This is the quantity the continuous
        ``alpha`` estimator consumes.
        """

        horizon = features.shape[0]
        hidden = self.hidden_activations(features)
        derivative = 1.0 - hidden * hidden
        rows = np.empty((horizon, self.n_actions * self.n_parameters))
        for k in range(horizon):
            blocks = []
            for action in range(self.n_actions):
                scale = self.W2[:, action] * derivative[k]
                gW1 = np.outer(features[k], scale)
                gb1 = scale
                gW2 = np.zeros((self.hidden, self.n_actions))
                gW2[:, action] = hidden[k]
                gb2 = np.zeros(self.n_actions)
                gb2[action] = 1.0
                blocks.append(
                    np.concatenate(
                        [gW1.ravel(), gb1.ravel(), gW2.ravel(), gb2.ravel()]
                    )
                )
            rows[k] = np.concatenate(blocks)
        return rows


def _ridge_gradients(head: MLPHead, snapshot, ridge: float):
    if ridge <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)
    W1, b1, W2, b2 = snapshot
    return (
        ridge * (head.W1 - W1),
        ridge * (head.b1 - b1),
        ridge * (head.W2 - W2),
        ridge * (head.b2 - b2),
    )


def _snapshot(head: MLPHead):
    return (head.W1.copy(), head.b1.copy(), head.W2.copy(), head.b2.copy())


def _gradient_norm(gradients) -> float:
    return float(
        np.sqrt(sum(float(np.sum(np.asarray(g) ** 2)) for g in gradients))
    )


class _MLPParameterization:
    """Mixin replacing the linear head with :class:`MLPHead`."""

    def _install_head(self, hidden: int, seed: int) -> None:
        self.head = MLPHead(
            feature_dim=self.features.shape[1],
            n_actions=self.n_actions,
            hidden=hidden,
            seed=seed,
        )
        # The inherited linear weights are unused; keep them zeroed so any
        # accidental reference is obvious rather than silently wrong.
        self.weights = np.zeros_like(self.weights)

    @property
    def policy(self) -> FloatArray:
        return _softmax(self.head.logits(self.features))

    def _descend(self, residual_fn) -> tuple[float, int]:
        """Shared projection loop; ``residual_fn()`` returns the (H, A) residual."""

        snapshot = _snapshot(self.head)
        norm = float("inf")
        steps = 0
        for steps in range(1, self.projection_steps + 1):
            gradients = self.head.parameter_gradient(
                self.features, residual_fn() / self.horizon
            )
            ridge = _ridge_gradients(self.head, snapshot, self.projection_ridge)
            gradients = tuple(g + r for g, r in zip(gradients, ridge))
            norm = _gradient_norm(gradients)
            if norm <= self.projection_tolerance:
                break
            self.head.step(gradients, self.projection_learning_rate)
        return norm, steps


class MLPProjectedGroupOMD(_MLPParameterization, ProjectedGroupOMD):
    """Uniform baseline under the non-linear parameterization."""

    def __init__(self, *, mlp_hidden: int = 32, mlp_seed: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._install_head(mlp_hidden, mlp_seed)

    def _apply_action_scores(
        self, action_scores: FloatArray, local_step_scales: FloatArray | None = None
    ) -> dict[str, float]:
        scores = np.asarray(action_scores, dtype=np.float64)
        scales = (
            np.ones(self.horizon)
            if local_step_scales is None
            else np.asarray(local_step_scales, dtype=np.float64)
        )
        old_policy = self.policy
        target_logits = np.log(np.maximum(old_policy, self.min_probability))
        target_logits += self.step_size * scales[:, None] * scores
        target_policy = _softmax(target_logits)
        before = self.head.flat_parameters()

        norm, steps = self._descend(lambda: self.policy - target_policy)

        new_policy = self.policy
        log_new = np.log(np.maximum(new_policy, self.min_probability))
        log_old = np.log(np.maximum(old_policy, self.min_probability))
        per_position_kl = np.sum(new_policy * (log_new - log_old), axis=1)
        log_target = np.log(np.maximum(target_policy, self.min_probability))
        projection_kl = (
            float(np.sum(target_policy * (log_target - log_new))) / self.horizon
        )
        return {
            "update_norm": float(np.linalg.norm(scales[:, None] * scores)),
            "kl_drift": float(per_position_kl.sum()),
            "max_position_kl": float(per_position_kl.max()),
            "parameter_update_norm": float(
                np.linalg.norm(self.head.flat_parameters() - before)
            ),
            "projection_kl": projection_kl,
            "projection_gradient_norm": norm,
            "projection_steps": float(steps),
        }


class MLPReliabilityWeightedProjectionOMD(
    _MLPParameterization, ReliabilityWeightedProjectionOMD
):
    """RWP-OMD under the non-linear parameterization.

    ``mixture_target`` is inherited verbatim, so the objective, the mixture, and
    the mean-normalized weights are identical to the linear case. Only the
    parameter family the mixture is projected into changes.
    """

    def __init__(self, *, mlp_hidden: int = 32, mlp_seed: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._install_head(mlp_hidden, mlp_seed)

    def _reliability_weighted_projection(
        self, action_scores: FloatArray, reliability: FloatArray
    ) -> dict[str, float]:
        scores = np.asarray(action_scores, dtype=np.float64)
        old_policy = self.policy
        _, mixture, normalized = self.mixture_target(action_scores, reliability)
        log_mixture = np.log(np.maximum(mixture, self.min_probability))
        before = self.head.flat_parameters()

        norm, steps = self._descend(
            lambda: normalized[:, None] * (self.policy - mixture)
        )

        new_policy = self.policy
        log_new = np.log(np.maximum(new_policy, self.min_probability))
        log_old = np.log(np.maximum(old_policy, self.min_probability))
        per_position_kl = np.sum(new_policy * (log_new - log_old), axis=1)
        target_kl = np.sum(mixture * (log_mixture - log_old), axis=1)
        projection_residual = np.sum(mixture * (log_mixture - log_new), axis=1)
        return {
            "update_norm": float(np.linalg.norm(normalized[:, None] * scores)),
            "kl_drift": float(per_position_kl.sum()),
            "max_position_kl": float(per_position_kl.max()),
            "parameter_update_norm": float(
                np.linalg.norm(self.head.flat_parameters() - before)
            ),
            "projection_kl": float(projection_residual.sum()) / self.horizon,
            "projection_gradient_norm": norm,
            "projection_steps": float(steps),
            "target_kl_total": float(target_kl.sum()),
            "realized_kl_total": float(per_position_kl.sum()),
            "projection_residual_total": float(projection_residual.sum()),
            "weight_spread": float(normalized.max() - normalized.min()),
        }
