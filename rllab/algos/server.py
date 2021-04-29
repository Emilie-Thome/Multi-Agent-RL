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
        self.theta_server = policy.get_param_values()
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


    # def obtain_samples(self, itr):
    #     paths_n = []
    #     for agent in self.agents:
    #         paths_n.append(agent.sampler.obtain_samples(itr))
    #     return paths_n

    # def process_samples(self, itr, paths_n):
    #     samples_data_n = []
    #     for paths, agent in zip(paths_n, self.agents):
    #         samples_data_n.append(agent.sampler.process_samples(itr, paths))
    #     return samples_data_n

    # def optimize_agents_policies(self, itr, samples_data_n):
    #     for samples_data, agent in zip(samples_data_n, self.agents):
    #         agent.optimize_policy(itr, samples_data)

    # @overrides
    # def optimize_policy(self):
    #     participants = self.generate_participants()

    #     delta_agents = self.collect_deltas(participants)
    #     for k, agent in enumerate(participants):
    #         self.transferred_bits += sys.getsizeof(delta_agents[k])

    #     delta_server = np.average(delta_agents, axis=0)
    #     for agent in self.agents:
    #         agent.server_update_mean_policy(delta_server)
    #         if agent in participants:
    #             agent.server_update_policy()

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
        for agent in self.agents:
            agent.GT_init(self.theta_server)

    @overrides
    def get_itr_snapshot(self, itr):
        return dict(
            itr=itr,
            policy=self.policy, #TODO : not necessary
            baseline=self.baseline,
            env=self.env,
        )

    @overrides
    def log_diagnostics(self, itr, paths_n):
        logger.log("saving snapshot...")
        params = self.get_itr_snapshot(itr)
        params["algo"] = self
        logger.record_tabular('TransfBits',self.transferred_bits)
        returns = [sum([rew*(self.discount**k) for k, rew in enumerate(path['rewards'])]) for paths in paths_n for path in paths]
        average_returns = np.mean(returns)
        logger.record_tabular('TotalAverageReturn', average_returns)
        logger.save_itr_params(itr, params)
        logger.log("saved")
        logger.dump_tabular(with_prefix=False)

    def generate_participants(self):
        agents = self.agents
        nb_participants = int(self.participation_rate*len(agents))
        participants = set()
        while len(participants) != nb_participants:
            participants.update({agents[random.randrange(len(agents))]})
        return participants

    def collect_deltas(self, participants):
        return [agent.transmit_to_server() for agent in participants]

    @overrides
    def train(self):
        self.start_worker()
        self.init_opt()
        for itr in range(self.current_itr, self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                participants = self.generate_participants()
                paths_n = []
                for agent in self.agents:
                    if (not itr % self.average_period) and (agent in participants):
                        agent.update_policy()
                    paths = agent.sampler.obtain_samples(itr)
                    paths_n.append(paths)
                    samples_data = agent.sampler.process_samples(itr, paths)
                    agent.GT_optimize(itr, samples_data)

                if itr and (not itr % self.average_period):                   
                    delta_agents = self.collect_deltas(participants)
                    self.transferred_bits += sum([sys.getsizeof(delta_agent) for delta_agent in delta_agents])
                    delta_server = np.average(delta_agents, axis=0)
                    self.theta_server = self.theta_server + self.learning_rate*self.gamma*delta_server
                    for agent in self.agents:
                        agent.transmit_to_agent(delta_server, self.theta_server)
                        agent.update_GT(self.average_period)

                self.log_diagnostics(itr, paths_n)
                self.current_itr = itr + 1

        if (self.n_itr - 1) % self.average_period != 0 :
            self.log_diagnostics(self.n_itr - 1, paths_n)

        self.shutdown_worker()