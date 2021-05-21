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

        self.grads_n = None
        self.p_traj_n = None
        self.observations_var = None
        self.actions_var = None
        self.returns_var = None
        self.epoch_policy_params = policy.get_param_values()
        self.gradient_approx = None
        self.variance_reduce_correction = None
        self.variance_reduce_gradient = None

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
        return [self.quantize_component(vector, component) for component in vector]

    def estimate_gradient(self, samples_data):# TODO: after that the agent send it to the server and receive the average of all agents estimation of the gradient
        inputs = ext.extract(
            samples_data,
            "observations", "actions", "returns"
        )
        gradients = []
        for index, grads in enumerate(self.grads_n):
            gradients.append([])
            for grad in grads:
                gradients[index] = gradients[index] + list(np.array(grad.eval({ self.observations_var: inputs[0],
                                                                                self.actions_var: inputs[1],
                                                                                self.returns_var: inputs[2]})).flat)
        self.gradient_approx = np.average(np.array(gradients), axis=0)

    def correction(self, samples_data):
        curr_policy_params = self.policy.get_param_values()
        inputs = ext.extract(
            samples_data,
            "observations", "actions", "returns"
        )

        gradients = []
        for index, grads in enumerate(self.grads_n):
            gradients.append([])
            for grad in grads:
                gradients[index] = gradients[index] + list(np.array(grad.eval({ self.observations_var: inputs[0],
                                                                                self.actions_var: inputs[1],
                                                                                self.returns_var: inputs[2]})).flat)
        curr_policy_params_gradients = np.average(np.array(gradients), axis=0)
        
        self.policy.set_param_values(self.epoch_policy_params)
        gradients = []
        for index, grads in enumerate(self.grads_n):
            gradients.append([])
            for grad in grads:
                gradients[index] = gradients[index] + list(np.array(grad.eval({ self.observations_var: inputs[0],
                                                                                self.actions_var: inputs[1],
                                                                                self.returns_var: inputs[2]})).flat)
        epoch_policy_params_gradients = np.average(np.array(gradients), axis=0, weights=self.importance_weights(samples_data))

        self.policy.set_param_values(curr_policy_params)
        self.variance_reduce_correction = np.array(curr_policy_params_gradients) - np.array(epoch_policy_params_gradients)

    # def importance_weights(self, observations, actions, dist_infos):
    #     # p(traj| policy_params) = mult_over_time(p(s_t+1|s_t,a_t)*policy_prob(a_t|s_t))
    #     # get_possible_next_states(self, state, action) returns a list of pairs (s', p(s'|s,a)))
    #     # But every return is a singlton and p(s'|s,a) = 1
    #     T = len(actions)
        
    #     curr_policy_params = self.policy.get_param_values()
    #     proba_curr_params = np.exp(sum([-(((actions[t]-dist_infos[t]["mean"])/np.exp(dist_infos[t]["log_std"]))**2+2*dist_infos[t]["log_std"]+np.log(2*np.pi))/2 for t in range(T)]))

    #     self.policy.set_param_values(self.epoch_policy_params)
    #     _, dist_infos = self.policy.get_actions(observations)
    #     proba_epoch_policy_params = np.exp(sum([-(((actions[t]-dist_infos[t]["mean"])/np.exp(dist_infos[t]["log_std"]))**2+2*dist_infos[t]["log_std"]+np.log(2*np.pi))/2 for t in range(T)]))

    #     self.policy.set_param_values(self.curr_policy_params)
    #     return proba_epoch_policy_params/proba_curr_params

    def importance_weights(self, samples_data):
        # p(traj| policy_params) = mult_over_time(p(s_t+1|s_t,a_t)*policy_prob(a_t|s_t))
        # get_possible_next_states(self, state, action) returns a list of pairs (s', p(s'|s,a)))
        # But every return is a singlton and p(s'|s,a) = 1
        inputs = ext.extract(
            samples_data,
            "observations", "actions"
        )    
        curr_policy_params = self.policy.get_param_values()
        proba_curr_params_n = np.array(self.p_traj_n(inputs[0], inputs[1]))
        self.policy.set_param_values(self.epoch_policy_params)
        proba_epoch_policy_params = np.array(self.p_traj_n(inputs[0], inputs[1]))
        self.policy.set_param_values(curr_policy_params)
        return proba_epoch_policy_params/proba_curr_params
        

    def gradient_init(self):
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
        
        traj_likelihood_n = dist.log_likelihood_sym(self.actions_var, dist_info_vars)
        self.p_traj_n = theano.function(inputs=[observations_var, actions_var], outputs=traj_likelihood_n, on_unused_input='ignore')
        
        # Note that we do not negate the objective, since we want to maximize the returns
        # surr = TT.mean(dist.log_likelihood_sym(self.actions_var, dist_info_vars) * self.returns_var)
        # Get the list of trainable parameters.
        # params = self.policy.get_params(trainable=True)
        # self.grads_n = theano.grad(surr, params)
        objective_n = self.returns_var[:,0]
        params = self.policy.get_params(trainable=True)
        self.grads_n = theano.grad(objective_n, params)


    def variance_reduction(self):
        self.variance_reduce_gradient =  self.gradient_approx + self.variance_reduce_correction

    def policy_update(self):
        agent_new_params = self.policy.get_param_values() + self.learning_rate * self.variance_reduce_gradient
        self.policy.set_param_values(agent_new_params)











    def update_policy_to_server(self):
        self.policy.set_param_values(self.theta_server)
        
    def compute_delta_agent(self):
        delta_agent = (self.theta_server - self.policy.get_param_values())/self.learning_rate
        self.delta_agent = self.quantize_vector(delta_agent) if self.quantize else delta_agent

    def transmit_to_server(self):
        return self.delta_agent

    def transmit_to_agent(self, delta_server, theta_server):
        self.delta_server = delta_server
        self.theta_server = theta_server