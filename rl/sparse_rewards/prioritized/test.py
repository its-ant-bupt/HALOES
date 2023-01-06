from typing import Dict, List
import os
import json

from tqdm import tqdm
import numpy as np
import gym

from rl.algorithms.PDDPG import Agent
from rl.environment.env import Environment

class Path(object):
    # x::Array{Float64} # x position [m]
    # y::Array{Float64} # y position [m]
    # yaw::Array{Float64} # yaw angle [rad]
    # a::Array{Float64} # acc [m/s2]
    # steer::Array{Float64} # steer [rad]
    def __init__(self, x, y, yaw, a, steer):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.a = a
        self.steer = steer


def test(agent, path_num, test_num):
    path_num = path_num
    env = Environment(path_num)
    OBS = env.reset(path_num)
    done = False
    while not done:
        state, curr_actgoal, curr_desgoal = OBS.values()
        assert len(curr_actgoal) == 3 and len(curr_desgoal) == 3, "Error!"
        obs = np.array(state)

        # Choose agent based action & make a transition
        action = agent.choose_action(obs)
        next_OBS, reward, done = env.step(action)

        next_state, next_actgoal, next_desgoal = next_OBS.values()
        assert len(next_actgoal) == 3 and len(next_desgoal) == 3, "Error!"
        next_obs = np.array(next_state)

        OBS = next_OBS




if __name__ == '__main__':

    # Init. Environment
    env = gym.make('FetchReach-v1')
    env.reset()

    # Init. Datapath
    data_path = os.path.abspath('sparse_rewards/prioritized/data')

    # Init. Testing
    n_games = 10
    test_data: List[Dict[str, np.ndarray]] = [] * n_games

    # Init. Agent
    agent = Agent(env=env, n_games=n_games, training=False)
    agent.load_models(data_path)

    for i in tqdm(range(n_games), desc=f'Testing', total=n_games):
        score_history: List[np.float32] = [] * n_games

        for _ in tqdm(range(n_games), desc=f'Testing', total=n_games):
            done: bool = False
            score: float = 0.0

            # Initial Reset of Environment
            OBS: Dict[str, np.array] = env.reset()

            while not done:
                # Unpack the observation
                state, curr_actgoal, curr_desgoal = OBS.values()
                obs = np.concatenate((state, curr_actgoal, curr_desgoal))

                # Choose agent based action & make a transition
                action = agent.choose_action(obs)
                next_OBS, reward, done, info = env.step(action)

                OBS = next_OBS
                score += reward

            score_history.append(score)

        print(f'Test Analysis:\n'
              f'Mean:{np.mean(score_history)}\n'
              f'Variance:{np.std(score_history)}')

        test_data.append({'Test Score': score_history})

    # Dump .json
    with open(os.path.join(data_path, 'testing_info.json'), 'w', encoding='utf8') as file:
        json.dump(test_data, file, indent=4, ensure_ascii=False)
