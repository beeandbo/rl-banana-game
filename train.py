"""A program for trianing a deep q learning agent

You can choose to tweak hyperparameters, control how long training lasts,
checkpoint to and from a file, etc.  Once you've finished trianing the
agent, you can test out final performance with eval.py

Example: pythonw train.py --loadfrom checkpoint.pth --saveto checkpoint2.pth --episodes 1000 --epsilon 0.1 --saveplot scores2.png
"""

import argparse
from unityagents import UnityEnvironment
import numpy as np
import coach
import agent
import time
import matplotlib.pyplot as plt

def moving_average(input, average_over):
    """Return a weighted moving average over the input.

    Params
    ======
        input (array_like): Weighted average is calculated over these elements
        average_over (integer): Window size of the weighted average
    """
    moving_average_ = 0;
    weight = 1./average_over
    output = []
    for elem in input:
        moving_average_ = moving_average_ * (1-weight) + elem * weight
        output.append(moving_average_)
    return output

def main():
    parser = argparse.ArgumentParser(description='Train a dqn agent to play the Unity Environment Banana app')
    parser.add_argument("--episodes", type=int, help="Number of training episodes to run", default=2000)
    parser.add_argument("--saveto", help="Save agent to this file after training", default='checkpoint.pth')
    parser.add_argument("--loadfrom", help="Load previously saved model before training")
    parser.add_argument("--min_score", type=float, help="Only save the model if the it achieves this score", default=13.)
    parser.add_argument("--epsilon", type=float, help="Starting epsilon", default=1.)
    parser.add_argument("--saveplot", help="Location to save plot of scores")


    args = parser.parse_args()

    env = UnityEnvironment(file_name="./Banana.app")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of actions
    action_size = brain.vector_action_space_size

    # examine the state space
    state = env_info.vector_observations[0]
    state_size = len(state)

    _agent = agent.Agent(state_size, action_size, seed=time.time(), epsilon_start=args.epsilon)
    if args.loadfrom:
        _agent.load(args.loadfrom)
        print("Loaded checkpoint: %s" % (args.loadfrom,))
    _coach = coach.Coach(_agent, env)

    scores = _coach.run_episodes(episodes=args.episodes, train=True)
    mean_score = np.mean(scores[-100:])
    if mean_score > 13.:
        if args.saveto:
            _agent.save(args.saveto)
        print("The training succeeded!")
    plt.plot(scores)
    plt.plot(moving_average(scores, 100), color='red')
    plt.ylabel('Episode scores')
    if args.saveplot:
        plt.savefig(args.saveplot, bbox_inches='tight')
    print("Your model achieved a final mean score of {}".format(mean_score))

main()
