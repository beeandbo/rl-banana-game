import argparse
from unityagents import UnityEnvironment
import numpy as np
import coach
import agent
import time
#from pyvirtualdisplay import Display

parser = argparse.ArgumentParser(description='Eval a dqn agent playing the Unity Environment Banana app')
parser.add_argument("--episodes", type=int, help="Number of training episodes to average over", default=100)
parser.add_argument("loadfrom", help="Model checkpoint used for eval")

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


_agent = agent.Agent(state_size, action_size, seed=time.time())
if args.loadfrom:
    _agent.load(args.loadfrom)
else:
    print(args.usage())
    exit()
_coach = coach.Coach(_agent, env)

scores = _coach.run_episodes(episodes=args.episodes, train=False)
mean_score = np.mean(scores[-100:])
print("Your model achieved a final mean score of {}".format(mean_score))
