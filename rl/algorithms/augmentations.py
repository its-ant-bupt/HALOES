from typing import Dict, List, Union

import numpy as np

from rl.algorithms import DDPG, PDDPG


def dense_reward(target: np.ndarray, state: np.ndarray) -> float:
    """Generates dense rewards as euclidean error norm of state and target vector

    Args:
        target (np.ndarray): target state vector of dimension (n)
        state (np.ndarray): state vector of dimension (m)

    Returns:
        float: reward
    """

    return -np.linalg.norm(target - state, ord=2)


def her_augmentation(agent: Union[DDPG.Agent, PDDPG.Agent],
                     states: List[Dict[str, np.ndarray]],
                     actions: List[np.ndarray],
                     next_states: List[Dict[str, np.ndarray]],
                     k: int = 1):
    """_summary_

    Args:
        agent (Union[DDPG.Agent, PDDPG.Agent]): _description_
        states (List[Dict[str, np.ndarray]]): _description_
        actions (List[np.ndarray]): _description_
        next_states (List[Dict[str, np.ndarray]]): _description_
        k (int, optional): _description_. Defaults to 8.
    """

    # Augment the replay buffer
    T = int(len(actions)/4)
    for index in range(T):
        for _ in range(k):
            # Always fetch index of upcoming episode transitions
            future = np.random.randint(index, T)

            # Unpack the buffers using the future index
            future_obs, future_actual_goal, _ = next_states[future].values()
            # her_augemented_goal = future_actual_goal

            # Compute HER Reward
            # reward = agent.env.compute_reward(her_augemented_goal, future_actual_goal, 1.0)
            reward = 10 + agent.env.reward_obstacle(future_obs, agent.env.obstacles)

            # Repack augmented episode transitions
            obs, _, _ = states[future].values()
            # state = np.concatenate((obs, future_actual_goal, her_augemented_goal))
            obs[3:6] = future_actual_goal
            state = np.array(obs)

            next_obs, _, _ = next_states[future].values()
            # next_state = np.concatenate((next_obs, future_actual_goal, her_augemented_goal))
            next_obs[3:6] = future_actual_goal
            next_state = np.array(next_obs)

            action = actions[future]

            # Add augmented episode transitions to agent's memory
            agent.memory.add(state, action, reward, next_state, True)
