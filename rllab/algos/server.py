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
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

class Server(BatchPolopt, Serializable):
    def __init__(self,
                 agents_number,
                 average_period,
                 participation_rate,
                 env,
                 policy,
                 snap_policy,
                 baseline,
                 N=100,
                 M=10,
                 T=100,
                 learning_rate=1e-3,
                 discount=0.99,
                 quantize=False,
                 quantization_tuning=4,
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
                           policy=GaussianMLPPolicy(env.spec, hidden_sizes=(8,),learn_std=False),
                           snap_policy=GaussianMLPPolicy(env.spec, hidden_sizes=(8,),learn_std=False),
                           N=N,
                           M=M,
                           T=T,
                           optimizer=optimizer,
                           baseline=baseline,
                           learning_rate=learning_rate,
                           discount=discount,
                           average_period=average_period,
                           quantize=quantize,
                           quantization_tuning=quantization_tuning, **kwargs)
                        for optimizer in optimizer]
        self.baseline = baseline

        self.nb_trajectories = n_itr
        self.learning_rate = learning_rate
        self.average_period = average_period

        self.participation_rate = participation_rate
        self.transferred_bits = 0

        super(Server, self).__init__(agents_number=agents_number,
                                    average_period=average_period,
                                    participation_rate=participation_rate,
                                    env=env,
                                    policy=policy,
                                    baseline=baseline,
                                    N=N,
                                    M=M,
                                    T=T,
                                    learning_rate=learning_rate,
                                    quantize=quantize,
                                    quantization_tuning=quantization_tuning,
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
            agent.init_SVRG()

    def log_diagnostics(self, itr, average_returns):
        logger.log("saving snapshot...")
        logger.record_tabular('TransfBits',self.transferred_bits)
        logger.record_tabular('TotalAverageReturn', average_returns)
        logger.log("saved")
        logger.dump_tabular(with_prefix=False)

    def generate_participants(self):
        agents = self.agents
        nb_participants = int(self.participation_rate*len(agents))
        participants = set()
        while len(participants) != nb_participants:
            participants.update({agents[random.randrange(len(agents))]})
        return participants

    def server_estimate_gradient(self):
        participants = self.generate_participants()
        gradients = []
        avg_returns = []
        for agent in self.agents:
            if (agent in participants):
                g, avg_return = agent.compute_and_communicate_gradient()
                gradients.append(g)
                avg_returns.append(avg_return)
        self.transferred_bits += sum([sys.getsizeof(g) for g in gradients])
        gradient_estimator = np.average(gradients, axis=0)
        for agent in self.agents:
            agent.receive_gradient_estimator(gradient_estimator)
        return np.mean(avg_returns)

    @overrides
    def train(self):
        self.start_worker()
        self.init_opt()
        j=0
        while j<s_tot-N:
            j+=N
            avg_return = self.server_estimate_gradient()
            self.log_diagnostics(self.current_itr, avg_return)


            

        for r in range(self.n_itr//self.average_period):
            avg_return = self.server_estimate_gradient()
            self.log_diagnostics(self.current_itr, avg_return)
            for c in range(self.average_period):
                self.current_itr = r*self.average_period + c
                for agent in self.agents:
                    agent.auto_train()
            for agent in self.agents:
                agent.snap_policy.set_param_values(agent.policy.get_param_values(trainable=True), trainable=True)    

        self.shutdown_worker()