from typing import Dict, Callable, Any, Sequence, Optional, List
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from rl.custom_types import LearningRateSchedule
from rl import Agent
from rl.simulator import SingleAgentWaitingSimulator
from rl.environments.bandit import random_bandit
from rl.agents import EpsilonGreedyRewardAveragingAgent
from rl.callbacks import AgentStateLogger


class BanditLearningCurves:
    """Bandit learning curves experiment.

    Calculates learning curves for the supplied agents. Stores results
    in the `results` attribute.

    Args:
      agent_builders: a dict with agent names as keys and callables
        that build the corresponding agents as values
      test_bed_size: number of random bandits to generate
      n_steps: length of simulations
      random_state: `None`, random seed or random number generator
    """

    def __init__(
        self,
        agent_builders: Dict[str, Callable[[], Agent]],
        test_bed_size: int,
        n_steps: int,
        random_state=None,
    ):
        n_levers = 10
        bandit_mean_params, bandit_sigma_params = (0.0, 1.0), (1.0, 0.0)
        results = []
        for bandit_id in range(test_bed_size):
            bandit = random_bandit(
                n_levers,
                mean_params=bandit_mean_params,
                sigma_params=bandit_sigma_params,
                random_state=random_state,
            )
            actions_ordered_by_value_desc = np.argsort(bandit.means)[::-1]
            action_ranks = np.argsort(actions_ordered_by_value_desc)
            for agent_name, agent_builder in agent_builders.items():
                agent = agent_builder()
                callbacks = [AgentStateLogger()]
                sim = SingleAgentWaitingSimulator(
                    bandit, agent, callbacks=callbacks
                )
                sim.run(n_steps)
                actions_taken_ranks = action_ranks[sim.history.actions]
                greedy_actions_ranks = action_ranks[
                    [np.argmax(s["Q"]) for s in callbacks[0].states]
                ]
                results.append(
                    {
                        "bandit_id": bandit_id,
                        "agent": agent_name,
                        "ranked_action_values": bandit.means[
                            actions_ordered_by_value_desc
                        ],
                        "action_ranks": actions_taken_ranks,
                        "actions_taken_optimal": (
                            (actions_taken_ranks == 0).astype(np.float_)
                        ),
                        "greedy_actions": greedy_actions_ranks,
                        "greedy_actions_optimal": (
                            (greedy_actions_ranks == 0).astype(np.float_)
                        ),
                        "rewards": np.array(sim.history.rewards),
                    }
                )
        self._results = results

    @property
    def results(self) -> List[Dict[str, Any]]:
        return self._results

    @property
    def average_optimal_action_value(self):
        """Mean optimal action value over the test bed."""
        return np.mean([r["ranked_action_values"][0] for r in self.results])

    def aggregated_results_summary(self) -> pd.DataFrame:
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
            pd.DataFrame(self.results)
            .groupby(["agent"])
            .agg(
                {
                    "rewards": "mean",
                    "actions_taken_optimal": "mean",
                    "greedy_actions_optimal": "mean",
                }
            )
            .explode(
                ["rewards", "actions_taken_optimal", "greedy_actions_optimal"]
            )
            .assign(t=lambda df: df.groupby(df.index).cumcount())
            .rename(
                {
                    "rewards": "mean_reward",
                    "actions_taken_optimal": "prob_action_taken_optimal",
                    "greedy_actions_optimal": "prob_greedy_action_optimal",
                },
                axis=1,
            )
            .reset_index()
        )

    def plot_results(self) -> None:
        """Plots aggregated results."""
        sns.set_style("whitegrid")
        results_summary = self.aggregated_results_summary()
        f, axs = plt.subplots(1, 3, figsize=(18, 4))
        plt.sca(axs[0])
        sns.lineplot(data=results_summary, x="t", y="mean_reward", hue="agent")
        plt.axhline(
            y=self.average_optimal_action_value,
            label="optimal",
            c="C3",
            ls=":",
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
        plt.axhline(y=1.0, label="optimal", c="C3", ls=":")
        plt.ylabel("% Optimal action taken")
        plt.legend()
        plt.sca(axs[2])
        sns.lineplot(
            data=results_summary,
            x="t",
            y="prob_greedy_action_optimal",
            hue="agent",
        )
        plt.axhline(y=1.0, label="optimal", c="C3", ls=":")
        plt.ylabel("% Optimal action identified")
        plt.legend()
        plt.show()


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