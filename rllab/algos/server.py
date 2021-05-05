import numpy as np
import random
import sys
import theano
import theano.tensor as TT
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab.algos.batch_polopt import BatchPolopt
from rllab.algos.agent import Agent
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.core.serializable import Serializable

class Server(BatchPolopt, Serializable):
    def __init__(self,
                 agents_number,
                 average_period,
                 participation_rate,
                 env,
                 policy,
                 baseline,
                 learning_rate=1e-3,
                 gamma=1,
                 difference_params=False,
                 quantize=False,
                 quantization_tuning=4,
                 optimizer=None,
                 optimizer_args=None,
                 whole_paths=False,
                 **kwargs):

        Serializable.quick_init(self, locals())
        if optimizer is None:
            default_args = dict(
                batch_size=None,
                max_epochs=1,
            )
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
            optimizer = [FirstOrderOptimizer(**optimizer_args)] * agents_number
        self.agents = [Agent(env=env,
                           policy=policy,
                           optimizer=optimizer,
                           baseline=baseline,
                           learning_rate=learning_rate,
                           average_period=average_period,
                           difference_params=difference_params,
                           quantize=quantize,
                           quantization_tuning=quantization_tuning,
                           whole_paths=whole_paths, **kwargs)
                        for optimizer in optimizer]
        self.baseline = baseline
        self.average_period = average_period
        self.participation_rate = participation_rate
        self.transferred_bits = 0
        self.whole_paths = whole_paths
        self.learning_rate = learning_rate
        self.gamma = gamma
        super(Server, self).__init__(agents_number=agents_number,
                                    average_period=average_period,
                                    participation_rate=participation_rate,
                                    env=env,
                                    policy=policy,
                                    baseline=baseline,
                                    learning_rate=learning_rate,
                                    gamma=gamma,
                                    difference_params=difference_params,
                                    quantize=quantize,
                                    quantization_tuning=quantization_tuning,
                                    optimizer=optimizer,
                                    optimizer_args=optimizer_args,
                                    whole_paths=whole_paths, **kwargs)

    @overrides
    def start_worker(self):
        for agent in self.agents:
            agent.start_worker()

    @overrides
    def shutdown_worker(self):
        for agent in self.agents:
            agent.shutdown_worker()

    @overrides
    def init_opt(self):
        default = self.policy.get_param_values()
        for agent in self.agents:
            agent.transmit_to_agent(default, default)
            agent.GT_init()

    @overrides
    def log_diagnostics(self, itr, paths_n, gradient_n):
        logger.log("saving snapshot...")
        logger.record_tabular('TransfBits',self.transferred_bits)
        returns = [sum([rew*(self.discount**k) for k, rew in enumerate(path['rewards'])]) for paths in paths_n for path in paths]
        average_returns = np.mean(returns)
        logger.record_tabular('TotalAverageReturn', average_returns)
        average_gradient = np.mean(gradient_n)
        logger.record_tabular('AverageGradient', average_gradient)
        logger.log("saved")
        logger.dump_tabular(with_prefix=False)

    def generate_participants(self):
        agents = self.agents
        nb_participants = int(self.participation_rate*len(agents))
        participants = set()
        while len(participants) != nb_participants:
            participants.update({agents[random.randrange(len(agents))]})
        return participants

    def compute_delta_server(self, participants):
        delta_agents = [agent.transmit_to_server() for agent in participants]
        self.transferred_bits += sum([sys.getsizeof(delta_agent) for delta_agent in delta_agents])
        return np.average(delta_agents, axis=0)
    
    def compute_theta_server(self, delta_server):
        return self.policy.get_param_values() - self.learning_rate*self.gamma*delta_server

    @overrides
    def train(self):
        self.start_worker()
        self.init_opt()
        for r in range(self.n_itr//self.average_period):
                participants = self.generate_participants()
                paths_n = []
                gradient_n = []
                for agent in self.agents:
                    if (agent in participants):
                        agent.update_policy_to_server()
                    for c in range(self.average_period):
                        with logger.prefix('r = '+str(r)+' | ' + 'c = '+str(c)+' | '):
                            self.current_itr = r*self.average_period + c
                            paths = agent.sampler.obtain_samples(self.current_itr)
                            samples_data = agent.sampler.process_samples(self.current_itr, paths)
                            gradient_estimator = agent.estimate_gradient(samples_data)
                            GT_based_estimator = agent.GT_based_estimator(gradient_estimator)
                            agent.GT_policy_update(GT_based_estimator)
                            if c == self.average_period-1 :
                                paths_n.append(paths)
                                gradient_n.append(np.average(gradient_estimator))
                            # if (c == 0) and (agent in participants):
                            #     returns = [sum([rew*(self.discount**k) for k, rew in enumerate(path['rewards'])]) for path in paths]
                            #     average_returns = np.mean(returns)
                            #     print('Agent updated Returns : ' + str(average_returns))
                            # if (c == 0) and (not agent in participants):
                            #     returns = [sum([rew*(self.discount**k) for k, rew in enumerate(path['rewards'])]) for path in paths]
                            #     average_returns = np.mean(returns)
                            #     print('Agent not updated Returns : ' + str(average_returns))
                            # if c == self.average_period - 1:
                            #     returns = [sum([rew*(self.discount**k) for k, rew in enumerate(path['rewards'])]) for path in paths]
                            #     average_returns = np.mean(returns)
                            #     print('Agent before potential update Returns : ' + str(average_returns))
                    agent.compute_delta_agent()

                delta_server = self.compute_delta_server(participants)
                theta_server = self.compute_theta_server(delta_server)
                self.policy.set_param_values(theta_server)

                for agent in self.agents:
                    agent.transmit_to_agent(delta_server, theta_server)
                    agent.GT_update()

                self.log_diagnostics(self.current_itr, paths_n, gradient_n)

        self.shutdown_worker()