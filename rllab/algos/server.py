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
        super(Server, self).__init__(agents_number=agents_number,
                                    average_period=average_period,
                                    participation_rate=participation_rate,
                                    env=env,
                                    policy=policy,
                                    baseline=baseline,
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
        for k, agent in enumerate(participants):
            self.transferred_bits += sys.getsizeof(delta_policy_params_n[k])

        delta_policy_params_mean = np.average(delta_policy_params_n, axis=0)
        for agent in self.agents:
            agent.server_update_mean_policy(delta_policy_params_mean)
            if agent in participants:
                agent.server_update_policy()

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
                for agent in self.agents:
                    if not itr % self.average_period: #TODO : if available for communications
                        agent.policy.set_param_values(server_params) #TODO: init server params

                    paths = agent.sampler.obtain_samples(itr)
                    print("###################")
                    print("###### paths ######")
                    print(paths)
                    print("###################")
                    print("###################")
                    is_recurrent = int(agent.policy.recurrent)

                    obs_var = agent.env.observation_space.new_tensor_variable(
                        'obs',
                        extra_dims=1 + is_recurrent,
                    )
                    action_var = agent.env.action_space.new_tensor_variable(
                        'action',
                        extra_dims=1 + is_recurrent,
                    )
                    advantage_var = ext.new_tensor(
                        'advantage',
                        ndim=1 + is_recurrent,
                        dtype=theano.config.floatX
                    )

                    state_info_vars = {
                        k: ext.new_tensor(
                            k,
                            ndim=2 + is_recurrent,
                            dtype=theano.config.floatX
                        ) for k in agent.policy.state_info_keys
                    }

                    state_info_vars_list = [state_info_vars[k] for k in agent.policy.state_info_keys]
                    
                    if is_recurrent:
                        valid_var = TT.matrix('valid')
                    else:
                        valid_var = None

                    dist_info_vars = agent.policy.dist_info_sym(obs_var, state_info_vars)
                    logli = dist.log_likelihood_sym(action_var, dist_info_vars)
                    # formulate as a minimization problem
                    # The gradient of the surrogate objective is the policy gradient
                    if is_recurrent:
                        surr_obj = - TT.sum(logli * advantage_var * valid_var) / TT.sum(valid_var)
                    else:
                        surr_obj = - TT.mean(logli * advantage_var)
                    input_list = [obs_var, action_var, advantage_var] + state_info_vars_list
                    grad = theano.grad(surr_obj, agent.policy.get_params(trainable=True), disconnected_inputs='ignore')
                    f_grad = theano.function(inputs=input_list,
                                        outputs=grad,
                                        on_unused_input='ignore')


                    samples_data = agent.sampler.process_samples(itr, paths)
                    inputs = ext.extract(
                        samples_data,
                        "observations", "actions", "advantages"
                    )
                    agent_infos = samples_data["agent_infos"]
                    state_info_list = [agent_infos[k] for k in agent.policy.state_info_keys]
                    inputs += tuple(state_info_list)
                    if agent.policy.recurrent:
                        inputs += (samples_data["valids"],)

                    dataset = BatchDataset(inputs, batch_size=32)
                    gradients = []
                    for batch in dataset.iterate(update=True):
                        gradients.append(f_grad(*batch))

                    gradient_estimator = np.average(gradients, axis=0)

                    GT_based_estimator = gradient_estimator - gradient_tracking # TODO : init gradient tracking

                    agent_new_params = agent.policy.get_params() + learning_rate * GT_based_estimator # TODO : instore learning rate
                    agent.policy.set_param_values(agent_new_params)

                    if itr and (not itr % self.average_period):
                        # TODO: agent sends to server the quantization
                        # TODO: agent updates 

                if itr and (not itr % self.average_period):
                    # TODO: server computes average agents quantizations and broadcast
                    # TODO: server updates its policy and broadcast

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