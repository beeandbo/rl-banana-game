from experience import Experience
import numpy as np
from collections import deque

class Coach():

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]

    def run_episode(self, train=True):
        env_info = self.env.reset(train_mode=train)[self.brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = self.agent.act(state, explore = train)
            env_info = self.env.step(action)[self.brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            if train:
                self.agent.learn(Experience(state, action, reward, next_state, done))
            state = next_state
            if done:
                break
        return score

    def diagnostic(self, episode, scores, average_scores_over):
        score_window = scores[-average_scores_over:]
        mean_score = np.mean(score_window)
        max_score = np.max(score_window)
        if (episode + 1) % 100 == 0:
            end = "\n"
        else:
            end = ""
        print("\rEpisode: {}, Mean: {}, Max: {}, Last: {}, Epsilon: {}".format(episode, mean_score, max_score, scores[-1], self.agent.get_epsilon()), end=end)

    def run_episodes(self, episodes = 2000, train = True, average_scores_over = 100):
        scores = []
        print("Train {}".format(train))
        for i in range(episodes):
            scores.append(self.run_episode(train))
            self.diagnostic(i, scores, average_scores_over)
        return scores
