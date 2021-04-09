import numpy as np
import io
import pickle
import zlib
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
                 server_env,
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
        self.agents = [Agent(env=server_env.agents_envs[k],
                           policy=policy,
                           optimizer=optimizer,
                           baseline=baseline, **kwargs)
                        for k, optimizer in enumerate(optimizer)]
        self.baseline = baseline
        self.average_period = average_period
        super(Server, self).__init__(agents_number=agents_number,
                                    average_period=average_period,
                                    env=server_env,
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

    def quantize(self, policy_params):
        bytes = io.BytesIO()
        pickle.dump(policy_params, bytes)
        return zlib.compress(bytes.getvalue())

    def dequantize(self, zbytes):
        bytes = zlib.decompress(zbytes)
        return pickle.loads(bytes)

    @overrides
    def optimize_policy(self):
        quantize = self.quantize
        dequantize = self.dequantize

        policy_compressedparams_n = [quantize(agent.policy.get_param_values()) for agent in self.agents]
        policy_decompressedparams_n = [dequantize(zbytes) for zbytes in policy_compressedparams_n]
        
        policy_params_mean = np.average(policy_decompressedparams_n, axis=0)
        policy_compressedparams_mean = quantize(policy_params_mean)
        for agent in self.agents:
            agent.policy.set_param_values(dequantize(policy_compressedparams_mean))

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
                    logger.save_itr_params(itr, params)
                    logger.log("saved")
                    logger.dump_tabular(with_prefix=False)
        if (self.n_itr - 1) % self.average_period != 0 :
            self.optimize_policy()
            logger.log("saving snapshot...")
            params = self.get_itr_snapshot(self.n_itr - 1)
            params["algo"] = self
            logger.save_itr_params(self.n_itr - 1, params)
            logger.log("saved")
            logger.dump_tabular(with_prefix=False)

        self.shutdown_worker()
        
        return np.mean([sum(path["rewards"])for paths in paths_n for path in paths])
