import numpy as np
import random
import sys
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
                 optimizer=None,
                 optimizer_args=None,
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
                           baseline=baseline, **kwargs)
                        for optimizer in optimizer]
        self.baseline = baseline
        self.average_period = average_period
        self.participation_rate = participation_rate
        self.transferred_bits = 0
        super(Server, self).__init__(agents_number=agents_number,
                                    average_period=average_period,
                                    participation_rate=participation_rate,
                                    env=env,
                                    policy=policy,
                                    baseline=baseline,
                                    optimizer=optimizer,
                                    optimizer_args=optimizer_args, **kwargs)


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
            agent.init_opt()

    def obtain_samples(self, itr):
        paths_n = []
        for agent in self.agents:
            paths_n.append(agent.sampler.obtain_samples(itr))
        return paths_n

    def process_samples(self, itr, paths_n):
        samples_data_n = []
        for paths, agent in zip(paths_n, self.agents):
            samples_data_n.append(agent.sampler.process_samples(itr, paths))
        return samples_data_n

    @overrides
    def log_diagnostics(self, paths_n):
        for paths, agent in zip(paths_n, self.agents):
            agent.log_diagnostics(paths)

    def optimize_agents_policies(self, itr, samples_data_n):
        for samples_data, agent in zip(samples_data_n, self.agents):
            agent.optimize_policy(itr, samples_data)

    def generate_participants(self):
        agents = self.agents
        nb_participants = int(self.participation_rate*len(agents))
        participants = set()
        while len(participants) != nb_participants:
            participants.update({agents[random.randrange(len(agents))]})
        return participants

    def collect_delta_policy_params(self, participants):
        return [agent.transmit_server() for agent in participants]

    @overrides
    def optimize_policy(self):
        participants = self.generate_participants()
        delta_policy_params_n = self.collect_delta_policy_params(participants)
        delta_policy_params_mean = np.average(delta_policy_params_n, axis=0)

        for k, agent in enumerate(participants):
            self.transferred_bits += sys.getsizeof(delta_policy_params_n[k])
            agent.server_update_policy(delta_policy_params_mean)

    @overrides
    def get_itr_snapshot(self, itr):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )

    @overrides
    def train(self):
        self.start_worker()
        self.init_opt()
        for itr in range(self.current_itr, self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                paths_n = self.obtain_samples(itr)
                samples_data_n = self.process_samples(itr, paths_n)
                self.log_diagnostics(paths_n)
                # print('Average Return:', np.mean([sum(path["rewards"])for paths in paths_n for path in paths]))
                self.optimize_agents_policies(itr, samples_data_n)
                if itr and (itr % self.average_period == 0):
                    self.optimize_policy()
                    logger.log("saving snapshot...")
                    params = self.get_itr_snapshot(itr)
                    self.current_itr = itr + 1
                    params["algo"] = self
                    logger.record_tabular('TransfBits',self.transferred_bits)
                    # print([str(rew) + '*' + str(self.discount) + '^' + str(k) for k, rew in enumerate(paths_n[0][0]['rewards'])])
                    returns = [sum([rew*(self.discount**k) for k, rew in enumerate(path['rewards'])]) for paths in paths_n for path in paths]
                    average_returns = np.mean(returns)
                    # print(average_returns)
                    # print(returns)
                    logger.record_tabular('TotalAverageReturn', average_returns)
                    logger.save_itr_params(itr, params)
                    logger.log("saved")
                    logger.dump_tabular(with_prefix=False)
        if (self.n_itr - 1) % self.average_period != 0 :
            self.optimize_policy()
            logger.log("saving snapshot...")
            params = self.get_itr_snapshot(self.n_itr - 1)
            params["algo"] = self
            logger.record_tabular('TransfBits',self.transferred_bits)
            average_returns = np.mean(np.array([path['returns'][0] for paths in paths_n for path in paths]))
            logger.record_tabular('TotalAverageReturn', np.mean(average_returns))
            logger.save_itr_params(self.n_itr - 1, params)
            logger.log("saved")
            logger.dump_tabular(with_prefix=False)

        self.shutdown_worker()
        
        return np.mean([sum(path["rewards"])for paths in paths_n for path in paths])