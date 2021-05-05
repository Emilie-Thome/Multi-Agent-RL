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
            average_period=1,
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
        self.difference_params = difference_params
        self.whole_paths = whole_paths
        self.learning_rate = learning_rate
        self.average_period = average_period

        self.gradient_tracking = None
        self.grads = None
        self.observations_var = None
        self.actions_var = None
        self.returns_var = None
        self.delta_agent = None
        self.delta_server = None
        self.theta_server = None

        self.opt_info = None
        super(Agent, self).__init__(env=env,
                                    policy=policy,
                                    baseline=baseline,
                                    quantize=quantize,
                                    learning_rate=learning_rate,
                                    quantization_tuning=quantization_tuning,
                                    whole_paths=whole_paths, **kwargs)

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
        print(vector)
        return [self.quantize_component(vector, component) for component in vector]

    def estimate_gradient(self, samples_data):
        inputs = ext.extract(
            samples_data,
            "observations", "actions", "returns"
        )
        gradients = []
        for g_t in self.grads:
            gradients = gradients + list(np.array(g_t.eval({self.observations_var: inputs[0],
                                                            self.actions_var: inputs[1],
                                                            self.returns_var: inputs[2]})).flat)
        return np.array(gradients)


    def GT_init(self):
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
        # Get the list of trainable parameters.
        params = self.policy.get_params(trainable=True)
        self.grads = theano.grad(surr, params)


        paths = self.sampler.obtain_samples(0)
        samples_data = self.sampler.process_samples(0, paths)
        gradient_estimator = self.estimate_gradient(samples_data)
        # TODO : this cause problem
        # self.gradient_tracking = np.array(gradients)
        self.gradient_tracking = np.zeros(gradient_estimator.size)

    def GT_update(self):
        self.gradient_tracking = self.gradient_tracking + (self.delta_agent - self.delta_server)/self.average_period
        
    def GT_based_estimator(self, gradient):
        return gradient - self.gradient_tracking

    def GT_policy_update(self, GT_based_estimator):
        agent_new_params = self.policy.get_param_values() + self.learning_rate * GT_based_estimator
        self.policy.set_param_values(agent_new_params)

    def update_policy_to_server(self):
        self.policy.set_param_values(self.theta_server)
        
    def compute_delta_agent(self):
        delta_agent = (self.theta_server - self.policy.get_param_values())/self.learning_rate
        self.delta_agent = self.quantize_vector(delta_agent)

    def transmit_to_server(self):
        return self.delta_agent

    def transmit_to_agent(self, delta_server, theta_server):
        self.delta_server = delta_server
        self.theta_server = theta_server