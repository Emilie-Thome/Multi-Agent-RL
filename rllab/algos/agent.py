import theano.tensor as TT
import theano
import random
import numpy as np
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab.misc import ext
from rllab.algos.batch_polopt import BatchPolopt
from rllab.optimizers.minibatch_dataset import BatchDataset
from rllab.core.serializable import Serializable


class Agent(BatchPolopt, Serializable):
    """
    Agent Policy Gradient.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            learning_rate=1e-3,
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
            optimizer = FirstOrderOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.quantize = quantize
        self.quantization_tuning = quantization_tuning
        self.opt_info = None
        self.theta_server = 0
        self.difference_params = difference_params
        self.whole_paths = whole_paths
        self.learning_rate = learning_rate
        self.gradient_tracking = None
        self.grads = None
        self.observations_var = None
        self.actions_var = None
        self.returns_var = None
        self.delta_agent = None
        self.delta_server = None

        super(Agent, self).__init__(env=env,
                                    policy=policy,
                                    baseline=baseline,
                                    quantize=quantize,
                                    learning_rate=learning_rate,
                                    quantization_tuning=quantization_tuning,
                                    whole_paths=whole_paths, **kwargs)

    def server_update_mean_policy(self, delta_policy_params):
        policy_params = self.theta_server - delta_policy_params if self.difference_params else delta_policy_params
        self.theta_server = policy_params

    def update_policy(self):
        self.policy.set_param_values(self.theta_server)

    def quantize_component(self, v, v_i):
        s = self.quantization_tuning
        norm_v = np.linalg.norm(v)
        abs_v_i = abs(v_i)
        a = abs_v_i/norm_v if norm_v else 0
        l = float(int(a*s))
        p = a*s - l
        rand = random.uniform(0,1)
        ksi_i = (l+1)/s if (rand < p) else l/s
        sgn_v_i = np.sign(v_i)
        return norm_v*sgn_v_i*ksi_i

    def quantize_vector(self, vector):
        return [self.quantize_component(vector, component) for component in vector]

    def transmit_to_server(self):
        self.delta_agent = (self.theta_server - self.policy.get_param_values())/self.learning_rate if self.difference_params else self.policy.get_param_values()
        if self.quantize:
            self.delta_agent = self.quantize_vector(self.delta_agent)
        return self.delta_agent

    def transmit_to_agent(self, delta_server, theta_server):
        self.delta_server = delta_server
        self.theta_server = theta_server


    def update_GT(self, average_period):
        self.gradient_tracking = self.gradient_tracking # + (self.delta_agent - self.delta_server)/average_period TODO : this cause problem


    def GT_init(self, theta_server):
        self.theta_server = theta_server

        # Create a Theano variable for storing the observations
        # We could have simply written `observations_var = TT.matrix('observations')` instead for this example. However,
        # doing it in a slightly more abstract way allows us to delegate to the environment for handling the correct data
        # type for the variable. For instance, for an environment with discrete observations, we might want to use integer
        # types if the observations are represented as one-hot vectors.
        self.observations_var = self.env.observation_space.new_tensor_variable(
            'observations',
            # It should have 1 extra dimension since we want to represent a list of observations
            extra_dims=1
        )
        self.actions_var = self.env.action_space.new_tensor_variable(
            'actions',
            extra_dims=1
        )
        self.returns_var = TT.vector('returns')

        # policy.dist_info_sym returns a dictionary, whose values are symbolic expressions for quantities related to the
        # distribution of the actions. For a Gaussian policy, it contains the mean and the logarithm of the standard deviation.
        dist_info_vars = self.policy.dist_info_sym(self.observations_var)

        # policy.distribution returns a distribution object under rllab.distributions. It contains many utilities for computing
        # distribution-related quantities, given the computed dist_info_vars. Below we use dist.log_likelihood_sym to compute
        # the symbolic log-likelihood. For this example, the corresponding distribution is an instance of the class
        # rllab.distributions.DiagonalGaussian
        dist = self.policy.distribution

        # Note that we do not negate the objective, since we want to maximize the returns
        surr = TT.mean(dist.log_likelihood_sym(self.actions_var, dist_info_vars) * self.returns_var)

        paths = self.sampler.obtain_samples(0)
        samples_data = self.sampler.process_samples(0, paths)
        inputs = ext.extract(
            samples_data,
            "observations", "actions", "returns"
        )

        # Get the list of trainable parameters.
        params = self.policy.get_params(trainable=True)
        self.grads = theano.grad(surr, params)
        gradients = []
        for g_t in self.grads:
            gradients = gradients + list(np.array(g_t.eval({self.observations_var: inputs[0],
                                                            self.actions_var: inputs[1],
                                                            self.returns_var: inputs[2]})).flat)
        # self.gradient_tracking = np.array(gradients) TODO : this cause problem
        self.gradient_tracking = np.array([0]*len(gradients))


    def GT_optimize(self, itr, samples_data):
        inputs = ext.extract(
            samples_data,
            "observations", "actions", "returns"
        )
        # Get the list of trainable parameters.
        gradients = []
        for g_t in self.grads:
            gradients = gradients + list(np.array(g_t.eval({self.observations_var: inputs[0],
                                                            self.actions_var: inputs[1],
                                                            self.returns_var: inputs[2]})).flat)
        gradient_estimator = np.array(gradients)
        GT_based_estimator = gradient_estimator - self.gradient_tracking
        print("_____________GT_based_estimator_____________")
        print(GT_based_estimator)
        agent_new_params = self.policy.get_param_values() + self.learning_rate * GT_based_estimator
        self.policy.set_param_values(agent_new_params)
        print("_____self.policy.get_param_values()_____")
        print(self.policy.get_param_values())

    # @overrides
    # def init_opt(self):
    #     is_recurrent = int(self.policy.recurrent)

    #     obs_var = self.env.observation_space.new_tensor_variable(
    #         'obs',
    #         extra_dims=1 + is_recurrent,
    #     )
    #     action_var = self.env.action_space.new_tensor_variable(
    #         'action',
    #         extra_dims=1 + is_recurrent,
    #     )
    #     advantage_var = ext.new_tensor(
    #         'advantage',
    #         ndim=1 + is_recurrent,
    #         dtype=theano.config.floatX
    #     )
    #     dist = self.policy.distribution
    #     old_dist_info_vars = {
    #         k: ext.new_tensor(
    #             'old_%s' % k,
    #             ndim=2 + is_recurrent,
    #             dtype=theano.config.floatX
    #         ) for k in dist.dist_info_keys
    #         }
    #     old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

    #     if is_recurrent:
    #         valid_var = TT.matrix('valid')
    #     else:
    #         valid_var = None

    #     state_info_vars = {
    #         k: ext.new_tensor(
    #             k,
    #             ndim=2 + is_recurrent,
    #             dtype=theano.config.floatX
    #         ) for k in self.policy.state_info_keys
    #     }
    #     state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

    #     dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
    #     logli = dist.log_likelihood_sym(action_var, dist_info_vars)
    #     kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

    #     # formulate as a minimization problem
    #     # The gradient of the surrogate objective is the policy gradient
    #     if is_recurrent:
    #         surr_obj = - TT.sum(logli * advantage_var * valid_var) / TT.sum(valid_var)
    #         mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
    #         max_kl = TT.max(kl * valid_var)
    #     else:
    #         surr_obj = - TT.mean(logli * advantage_var)
    #         mean_kl = TT.mean(kl)
    #         max_kl = TT.max(kl)

    #     input_list = [obs_var, action_var, advantage_var] + state_info_vars_list
    #     if is_recurrent:
    #         input_list.append(valid_var)

    #     self.optimizer.update_opt(surr_obj, target=self.policy, inputs=input_list)

    #     f_kl = ext.compile_function(
    #         inputs=input_list + old_dist_info_vars_list,
    #         outputs=[mean_kl, max_kl],
    #     )
    #     self.opt_info = dict(
    #         f_kl=f_kl,
    #     )


    # @overrides
    # def optimize_policy(self, itr, samples_data):
    #     inputs = ext.extract(
    #         samples_data,
    #         "observations", "actions", "advantages"
    #     )
    #     agent_infos = samples_data["agent_infos"]
    #     state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
    #     inputs += tuple(state_info_list)
    #     if self.policy.recurrent:
    #         inputs += (samples_data["valids"],)
    #     dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
    #     self.optimizer.optimize(inputs)

