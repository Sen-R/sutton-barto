from typing import Dict, Callable, Any, Sequence, Optional, List, Union
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from tqdm.notebook import tqdm  # type: ignore
from rl.custom_types import LearningRateSchedule
from rl import Agent
from rl.simulator import SingleAgentWaitingSimulator
from rl.environments.bandit import random_bandit
from rl.agents import EpsilonGreedyRewardAveragingAgent
from rl.callbacks import History, AgentStateLogger, EnvironmentStateLogger


class BanditResults:
    """Results of bandit experiment.

    Returned by `bandit_experiment`, contains methods for analysing
    and visualising results, as well as for loading and saving them.
    """

    def __init__(self, results: List[Dict[str, Any]], logging_period: int = 1):
        self._results = results
        self.logging_period = logging_period

    @property
    def raw(self) -> List[Dict[str, Any]]:
        return self._results

    def save(self, file_path: Union[str, os.PathLike]):
        save_path = Path(file_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "results": self.raw,
                    "logging_period": self.logging_period,
                },
                f,
            )

    @classmethod
    def load(cls, file_path: Union[str, bytes, os.PathLike]):
        with open(file_path, "rb") as f:
            return cls(**pickle.load(f))

    @property
    def average_optimal_action_value(self):
        """Mean optimal action value over the test bed."""
        optimal_action_values = np.stack(
            [r["optimal_action_values"] for r in self.results]
        )
        return np.mean(optimal_action_values, axis=0)

    def summary(self) -> pd.DataFrame:
        """Summarises results and returns as a DataFrame.

        Returned DataFrame contains the following fields:
        - `agent`: agent ID
        - `t`: time_step (zero-indexed)
        - `mean_reward`: average reward for this agent and time step
        - `prob_action_taken_optimal`: fraction of simulations when the
          optimal action was taken by the agent at this time step
        - `prob_greedy_action_optimal`: fraction of simulations when then
          greedy action (according to agent's Q estimates) was actually
          the optimal action.
        """
        return (
            pd.DataFrame(self.raw)
            .groupby(["agent"])
            .agg(
                {
                    "rewards": "mean",
                    "actions_taken_optimal": "mean",
                    "greedy_actions_optimal": "mean",
                    "optimal_action_values": "mean",
                }
            )
            .explode(
                [
                    "rewards",
                    "actions_taken_optimal",
                    "greedy_actions_optimal",
                    "optimal_action_values",
                ]
            )
            .assign(
                t=lambda df: (
                    self.logging_period * (1 + df.groupby(df.index).cumcount())
                )
            )
            .rename(
                {
                    "rewards": "mean_reward",
                    "actions_taken_optimal": "prob_action_taken_optimal",
                    "greedy_actions_optimal": "prob_greedy_action_optimal",
                    "optimal_action_values": "mean_optimal_action_value",
                },
                axis=1,
            )
            .reset_index()
        )

    def plot(self) -> None:
        """Plots aggregated results."""
        sns.set_style("whitegrid")
        results_summary = self.summary()
        f, axs = plt.subplots(1, 3, figsize=(18, 4))
        plt.sca(axs[0])
        sns.lineplot(data=results_summary, x="t", y="mean_reward", hue="agent")
        plt.plot(
            results_summary["t"],
            results_summary["mean_optimal_action_value"],
            c="C3", alpha=0.3,
            label="optimal",
        )
        plt.ylabel("Average reward")
        plt.legend()
        plt.sca(axs[1])
        sns.lineplot(
            data=results_summary,
            x="t",
            y="prob_action_taken_optimal",
            hue="agent",
        )
        plt.axhline(y=1.0, label="optimal", c="C3", alpha=0.3)
        plt.ylabel("% Optimal action taken")
        plt.legend()
        plt.sca(axs[2])
        sns.lineplot(
            data=results_summary,
            x="t",
            y="prob_greedy_action_optimal",
            hue="agent",
        )
        plt.axhline(y=1.0, label="optimal", c="C3", alpha=0.3)
        plt.ylabel("% Optimal action identified")
        plt.legend()
        plt.show()


def bandit_experiment(
    agent_builders: Dict[str, Callable[[], Agent]],
    test_bed_size: int,
    n_steps: int,
    *,
    logging_period: int = 1,
    random_state=None,
) -> BanditResults:
    """Bandit learning curves experiment.

    Calculates learning curves for the supplied agents

    Args:
      agent_builders: a dict with agent names as keys and callables
        that build the corresponding agents as values
      test_bed_size: number of random bandits to generate
      n_steps: length of simulations
      random_state: `None`, random seed or random number generator

    Returns:
      `BanditResults` object containing results
    """
    n_levers = 10
    bandit_mean_params, bandit_sigma_params = (0.0, 1.0), (1.0, 0.0)
    results = []
    for bandit_id in tqdm(range(test_bed_size)):
        bandit = random_bandit(
            n_levers,
            mean_params=bandit_mean_params,
            sigma_params=bandit_sigma_params,
            random_state=random_state,
        )
        for agent_name, agent_builder in agent_builders.items():
            agent = agent_builder()
            history = History(logging_period)
            agent_state_log = AgentStateLogger(logging_period)
            bandit_state_log = EnvironmentStateLogger(logging_period)
            callbacks = [history, agent_state_log, bandit_state_log]
            sim = SingleAgentWaitingSimulator(
                bandit, agent, callbacks=callbacks
            )
            sim.run(n_steps)
            action_value_matrix = np.stack(
                [s["means"] for s in bandit_state_log.states]
            )
            optimal_actions = np.argmax(action_value_matrix, axis=-1)
            optimal_action_values = np.squeeze(
                np.take_along_axis(
                    action_value_matrix,
                    optimal_actions[:, np.newaxis],
                    axis=-1
                )
            )
            actions = np.array(history.actions)
            greedy_actions = np.array(
                [np.argmax(s["Q"]) for s in agent_state_log.states]
            )
            results.append(
                {
                    "bandit_id": bandit_id,
                    "agent": agent_name,
                    "optimal_action_values": optimal_action_values,
                    "actions": actions,
                    "actions_taken_optimal": (
                        (actions == optimal_actions).astype(np.float_)
                    ),
                    "greedy_actions": greedy_actions,
                    "greedy_actions_optimal": (
                        (greedy_actions == optimal_actions).astype(np.float_)
                    ),
                    "rewards": np.array(history.rewards),
                }
            )
    return BanditResults(results, logging_period=logging_period)


def get_epsilon_greedy_bandit_agent_builder(
    epsilon: float,
    n_actions: int,
    *,
    initial_action_values: Optional[Sequence[float]] = None,
    learning_rate_schedule: Optional[LearningRateSchedule] = None,
    random_state=None,
) -> Callable[[], Agent]:
    """Returns a callable that builds (when called) the specified agent."""

    def agent_builder() -> Agent:
        return EpsilonGreedyRewardAveragingAgent(
            epsilon,
            n_actions,
            initial_action_values=initial_action_values,
            learning_rate_schedule=learning_rate_schedule,
            random_state=random_state,
        )

    return agent_builder
