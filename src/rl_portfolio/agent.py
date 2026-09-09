"""Stable-Baselines3 PPO wrapper."""
from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .config import PPO_PARAMS


def build_ppo(env, *, seed: int = 1986, verbose: int = 0, device: str = "auto") -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=verbose,
        device=device,
        **PPO_PARAMS,
    )


class EpisodeLoggerCallback(BaseCallback):
    """Store episode reward and terminal portfolio value."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_endvalues: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_endvalues.append(float(info.get("portfolio_value", float("nan"))))
        return True
