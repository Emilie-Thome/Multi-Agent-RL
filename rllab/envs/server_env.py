from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
import numpy as np
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


class AgentEnv(Env):

    @autoargs.arg("lower_bound", type=int,
                  help="Observation lower bound")
    @autoargs.arg("upper_bound", type=int,
                  help="Observation upper bound")
    def __init__(self, lower_bound=-np.inf, upper_bound=np.inf):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @property
    @overrides
    def observation_space(self):
        return Box(low=self.lower_bound, high=self.upper_bound, shape=(2,))

    @property
    @overrides
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

    @overrides
    def reset(self):
        self._state = np.random.uniform(-1, 1, size=(2,))
        observation = np.copy(self._state)
        return observation

    @overrides
    def step(self, action):
        self._state = self._state + action
        x, y = self._state
        reward = - (x**2 + y**2) ** 0.5
        done = abs(x) < 0.01 and abs(y) < 0.01
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    @overrides
    def render(self):
        print 'current state:', self._state

class ServerEnv(Env):

    @autoargs.arg("agents_number", type=int,
                  help="Number of agents")
    def __init__(self, agents_number=1, lower_bound=-np.inf, upper_bound=np.inf):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.agents_number = agents_number
        partition = (upper_bound-lower_bound)/agents_number
        self.agents_envs = [AgentEnv(lower_bound + partition*i, lower_bound + partition*(i+1)) for i in range(agents_number)]

    @property
    @overrides
    def observation_space(self):
        return Box(low=self.lower_bound, high=self.upper_bound, shape=(self.agents_number,2))

    @property
    @overrides
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(self.agents_number,2))

    @overrides
    def reset(self):
        return np.array([env.reset() for env in self.agents_envs])

    @overrides
    def step(self, action):
    	next_observation = []
    	rewards = []
    	done = False
    	for k, env in enumerate(self.agents_envs) :
    		agent_next_observation, agent_reward, agent_done, _ = env.step(action[k])
    		next_observation.append(agent_next_observation)
    		rewards.append(agent_reward)
    		done = done or agent_done
        return Step(observation=np.array(next_observation), reward=sum(rewards), done=done)

    @overrides
    def render(self):
        print 'current state:', self._state