from typing import Dict, Callable, Any, List, Union, Optional
import os
from pathlib import Path
import pickle
from itertools import product
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.ticker import PercentFormatter  # type: ignore
import seaborn as sns  # type: ignore
from joblib import Parallel, delayed  # type: ignore
from rl import Agent
from rl.simulator import SingleAgentWaitingSimulator
from rl.environments.bandit import MultiArmedBandit
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
        results_summary.groupby("t")["mean_optimal_action_value"].mean().plot(
            c="C3",
            alpha=0.3,
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
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
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
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.legend()
        plt.show()


def bandit_experiment(
    agent_builders: Dict[str, Callable[..., Agent]],
    bandit_builder: Callable[..., MultiArmedBandit],
    test_bed_size: int,
    n_steps: int,
    *,
    logging_period: int = 1,
    entropy: Optional[int] = None,
    n_jobs: Optional[int] = None,
    verbose: int = 0,
    results_file: Optional[Union[str, os.PathLike]] = None,
) -> BanditResults:
    """Bandit learning curves experiment.

    Calculates learning curves for the supplied agents.

    Args:
      agent_builders: a dict with agent names as keys and callables
        that build the corresponding agents as values
      bandit_builder: a callable that creates a fresh bandit environment to
        the specifications required for the current experiment
      test_bed_size: number of random bandits to generate
      n_steps: length of simulations
      logging_period: how frequently to log results
      entropy: source of entropy to generate random seeds for agents and
        environments
      n_jobs: how many jobs to distribute simulations over (use -1 for
        maximum)
      verbose: verbosity of `joblib` progress meter
      results_file: if set, the file to load results from (if it exists) or
        cache results to (if it doesn't)

    Returns:
      `BanditResults` object containing results
    """

    def loop_fn(
        bandit_id: int,
        bandit_builder: Callable[..., MultiArmedBandit],
        agent_name: str,
        agent_builder: Callable[..., Agent],
        n_steps: int,
        logging_period: int,
        seed: int,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        bandit = bandit_builder(random_state=rng)
        agent = agent_builder(random_state=rng)
        history = History(logging_period)
        agent_state_log = AgentStateLogger(logging_period)
        bandit_state_log = EnvironmentStateLogger(logging_period)
        callbacks = [history, agent_state_log, bandit_state_log]
        sim = SingleAgentWaitingSimulator(bandit, agent, callbacks=callbacks)
        sim.run(n_steps)
        actions = np.array(history.actions)
        action_value_matrix = np.stack(
            [s["means"] for s in bandit_state_log.states]
        )
        optimal_action_values = np.max(action_value_matrix, axis=-1)
        taken_action_values = np.squeeze(
            np.take_along_axis(
                action_value_matrix, actions[:, np.newaxis], axis=-1
            )
        )
        greedy_actions = np.array(
            [np.argmax(s["Q"]) for s in agent_state_log.states]
        )
        greedy_action_values = np.squeeze(
            np.take_along_axis(
                action_value_matrix,
                greedy_actions[:, np.newaxis],
                axis=-1,
            )
        )
        return {
            "bandit_id": bandit_id,
            "agent": agent_name,
            "optimal_action_values": optimal_action_values,
            "actions": actions,
            "actions_taken_optimal": (
                (taken_action_values == optimal_action_values).astype(
                    np.float_
                )
            ),
            "greedy_actions": greedy_actions,
            "greedy_actions_optimal": (
                (greedy_action_values == optimal_action_values).astype(
                    np.float_
                )
            ),
            "rewards": np.array(history.rewards),
        }

    if results_file:
        try:
            return BanditResults.load(results_file)
        except FileNotFoundError:
            print(
                "results_file not found, "
                f"running experiment afresh: {results_file}"
            )

    seed_sq = np.random.SeedSequence(entropy).generate_state(
        test_bed_size * len(agent_builders)
    )
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(loop_fn)(
            bandit_id,
            bandit_builder,
            agent_name,
            agent_builder,
            n_steps,
            logging_period,
            seed,
        )
        for seed, (bandit_id, (agent_name, agent_builder)) in zip(
            seed_sq, product(range(test_bed_size), agent_builders.items())
        )
    )
    bandit_results = BanditResults(results, logging_period=logging_period)

    if results_file:
        bandit_results.save(results_file)
    return bandit_results
