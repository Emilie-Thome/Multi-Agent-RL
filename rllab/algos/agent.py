import theano.tensor as TT
import theano
import random
import numpy as np
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab.misc import ext
from rllab.algos.batch_polopt import BatchPolopt
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
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
            difference_params=False,
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
            optimizer = FirstOrderOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.quantize = quantize
        self.quantization_tuning = quantization_tuning
        self.opt_info = None
        self.policy_params_last_update = 0
        self.difference_params = difference_params
        super(Agent, self).__init__(env=env,
                                    policy=policy,
                                    baseline=baseline,
                                    quantize=quantize,
                                    quantization_tuning=quantization_tuning, **kwargs)

    def server_update_policy(self, delta_policy_params):
        policy_params = self.policy_params_last_update - delta_policy_params if self.difference_params else delta_policy_params
        self.policy.set_param_values(policy_params)
        self.policy_params_last_update = policy_params

    def quantize_component(self, v, v_i):
        s = self.quantization_tuning
        norm_v = np.linalg.norm(v)
        abs_v_i = abs(v_i)
        a = abs_v_i/norm_v
        l = int(a*s)
        p = a*s - l
        rand = random.uniform(0,1)
        ksi_i = (l+1)/s if (rand < p) else l/s
        sgn_v_i = np.sign(v_i)
        return norm_v*sgn_v_i*ksi_i

    def quantize_vector(self, vector):
        return [self.quantize_component(vector, component) for component in vector]

    def transmit_server(self):
        print("#################################################")
        print("########### policy_params_last_update ###########")
        print(self.policy_params_last_update)
        print("#################################################")
        print("#################################################")
        to_send = self.policy_params_last_update - self.policy.get_param_values() if self.difference_params else self.policy.get_param_values()
        if self.quantize:
            to_send = self.quantize_vector(to_send)
        return to_send

    @overrides
    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)

        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        dist = self.policy.distribution
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None

        state_info_vars = {
            k: ext.new_tensor(
                k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in self.policy.state_info_keys
        }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        logli = dist.log_likelihood_sym(action_var, dist_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        if is_recurrent:
            surr_obj = - TT.sum(logli * advantage_var * valid_var) / TT.sum(valid_var)
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            max_kl = TT.max(kl * valid_var)
        else:
            surr_obj = - TT.mean(logli * advantage_var)
            mean_kl = TT.mean(kl)
            max_kl = TT.max(kl)

        input_list = [obs_var, action_var, advantage_var] + state_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(surr_obj, target=self.policy, inputs=input_list)

        f_kl = ext.compile_function(
            inputs=input_list + old_dist_info_vars_list,
            outputs=[mean_kl, max_kl],
        )
        self.opt_info = dict(
            f_kl=f_kl,
        )

    @overrides
    def optimize_policy(self, itr, samples_data):
        inputs = ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        )
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        inputs += tuple(state_info_list)
        if self.policy.recurrent:
            inputs += (samples_data["valids"],)
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        self.optimizer.optimize(inputs)