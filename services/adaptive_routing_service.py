"""
RL-inspired adaptive provider routing (contextual bandit style).
"""
import logging
import random
import threading
from typing import Dict, List

from core.config import settings

logger = logging.getLogger(__name__)


class AdaptiveRoutingService:
    """
    Maintains per-framework provider rewards and reorders provider chain.
    Uses epsilon-greedy exploration + EMA reward updates.
    """

    def __init__(self):
        self.enabled = bool(settings.ENABLE_ADAPTIVE_ROUTING)
        self.epsilon = max(0.0, min(float(settings.ADAPTIVE_ROUTING_EPSILON or 0.08), 0.5))
        self.alpha = max(0.01, min(float(settings.ADAPTIVE_ROUTING_ALPHA or 0.25), 0.9))
        self.target_latency_ms = max(400, int(settings.ADAPTIVE_ROUTING_TARGET_LATENCY_MS or 3500))
        self._state: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._lock = threading.Lock()
        self._rng = random.Random()

    @staticmethod
    def _default_provider_stats() -> Dict[str, float]:
        return {
            "reward": 0.5,
            "samples": 0.0,
            "successes": 0.0,
        }

    def get_preferred_chain(self, framework: str, default_chain: List[str]) -> List[str]:
        chain = list(dict.fromkeys(default_chain or []))
        if not self.enabled or len(chain) <= 1:
            return chain

        safe_framework = (framework or "general").strip().lower()
        with self._lock:
            framework_state = self._state.setdefault(safe_framework, {})
            for provider in chain:
                framework_state.setdefault(provider, self._default_provider_stats().copy())

            if self._rng.random() < self.epsilon:
                randomized = list(chain)
                self._rng.shuffle(randomized)
                logger.info(
                    "Adaptive routing explore framework=%s chain=%s",
                    safe_framework,
                    ",".join(randomized),
                )
                return randomized

            ranked = sorted(
                chain,
                key=lambda provider: (
                    framework_state.get(provider, {}).get("reward", 0.5),
                    framework_state.get(provider, {}).get("successes", 0.0),
                    framework_state.get(provider, {}).get("samples", 0.0),
                ),
                reverse=True,
            )
            logger.info(
                "Adaptive routing exploit framework=%s chain=%s",
                safe_framework,
                ",".join(ranked),
            )
            return ranked

    def record_outcome(
        self,
        framework: str,
        provider: str,
        validation_passed: bool,
        confidence_score: float,
        latency_ms: int,
    ):
        if not self.enabled or not provider:
            return

        safe_framework = (framework or "general").strip().lower()
        safe_provider = (provider or "").strip().lower()
        confidence = max(0.0, min(float(confidence_score or 0.0), 1.0))
        latency = max(1, int(latency_ms or 1))

        # Reward composition: validation dominates, then confidence, then latency.
        validation_reward = 1.0 if validation_passed else 0.0
        latency_reward = max(0.0, 1.0 - (latency / float(self.target_latency_ms)))
        reward = (validation_reward * 0.6) + (confidence * 0.3) + (latency_reward * 0.1)

        with self._lock:
            framework_state = self._state.setdefault(safe_framework, {})
            stats = framework_state.setdefault(safe_provider, self._default_provider_stats().copy())
            old_reward = stats["reward"]
            stats["reward"] = (1.0 - self.alpha) * old_reward + self.alpha * reward
            stats["samples"] += 1.0
            if validation_passed:
                stats["successes"] += 1.0

            logger.info(
                "Adaptive routing update framework=%s provider=%s reward=%.4f old_reward=%.4f samples=%s",
                safe_framework,
                safe_provider,
                stats["reward"],
                old_reward,
                int(stats["samples"]),
            )

    def get_state_snapshot(self) -> Dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "target_latency_ms": self.target_latency_ms,
                "frameworks": self._state.copy(),
            }


adaptive_routing_service = AdaptiveRoutingService()
