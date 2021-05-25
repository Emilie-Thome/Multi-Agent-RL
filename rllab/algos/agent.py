import theano.tensor as TT
import theano
import numpy as np
import random
from lasagne.updates import sgd
from rllab.sampler import parallel_sampler
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab.misc import ext
from rllab.algos.batch_polopt import BatchPolopt
from rllab.optimizers.minibatch_dataset import BatchDataset
from rllab.core.serializable import Serializable


def unpack(i_g):
    i_g_arr = [np.array(x) for x in i_g]
    res = i_g_arr[0].reshape(i_g_arr[0].shape[0]*i_g_arr[0].shape[1])
    res = np.concatenate((res,i_g_arr[1]))
    res = np.concatenate((res,i_g_arr[2][0]))
    res = np.concatenate((res,i_g_arr[3]))
    return res

class Agent(BatchPolopt, Serializable):
    """
    Agent Policy Gradient.
    """

    def __init__(
            self,
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
            optimizer = FirstOrderOptimizer(**optimizer_args)
        self.optimizer = optimizer

        self.quantize = quantize
        self.quantization_tuning = quantization_tuning

        self.policy = policy
        self.snap_policy = snap_policy

        self.gradient_approx = None
        self.f_train = None
        self.f_update = None
        self.f_importance_weights = None
        self.f_train_SVRG = None

        self.N = N # N trajectories per iteration
        self.M = M # M secondary trajectories
        self.T = T # each trajectory will have at most T time steps
        self.learning_rate = learning_rate

        super(Agent, self).__init__(env=env,
                                    policy=policy,
                                    snap_policy=snap_policy,
                                    baseline=baseline,
                                    optimizer=optimizer,
                                    quantize=quantize,
                                    quantization_tuning=quantization_tuning,
                                    learning_rate=learning_rate,
                                    discount=discount, **kwargs)

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

    def quantize_gradient(self, gradient):
        to_vector = lambda arr : arr.reshape(-1)
        quant = lambda component : self.quantize_component(np.concatenate(map(to_vector, gradient)), component)
        
        quant_grad_0 = np.array([map(quant, grad) for grad in gradient[0]])
        quant_grad_1 = np.array(map(quant, gradient[1]))
        quant_grad_2 = np.array([map(quant, grad) for grad in gradient[2]])
        quant_grad_3 = np.array(map(quant, gradient[3]))

        return [quant_grad_0, quant_grad_1, quant_grad_2, quant_grad_3]

    def compute_and_communicate_gradient(self):
        paths = parallel_sampler.sample_paths_on_trajectories(self.snap_policy.get_param_values(),
                                                            self.N,
                                                            self.T,
                                                            show_bar=False)
        observations = [p["observations"] for p in paths]
        actions = [p["actions"] for p in paths]
        d_rewards = [p["rewards"] for p in paths]

        temp = list()
        for x in d_rewards:
            z=list()
            t=1
            for y in x:
                z.append(y*t)
                t*=self.discount
            temp.append(np.array(z))
        d_rewards=temp

        gradient = self.f_train(observations[0], actions[0], d_rewards[0])
        gradient_fv = [unpack(gradient)]
        for ob,ac,rw in zip(observations[1:],actions[1:],d_rewards[1:]):
            i_g = self.f_train(ob, ac, rw)
            gradient_fv.append(unpack(i_g))
            gradient = [sum(x) for x in zip(gradient,i_g)]
        gradient = [x/len(paths) for x in gradient]

        rewards_snapshot = np.array([sum(p["rewards"]) for p in paths])
        avg_return = np.mean([sum(p["rewards"]) for p in paths])  
        print('Average Return:', avg_return)

        if self.quantize :
            return self.quantize_gradient(gradient), avg_return
        else :
            return gradient, avg_return

    def receive_gradient_estimator(self, gradient):
        self.gradient_approx = gradient
        self.f_update(gradient[0],gradient[1],gradient[2],gradient[3])

    def auto_train(self):
        sub_paths = parallel_sampler.sample_paths_on_trajectories(self.snap_policy.get_param_values(),
                                                                self.M,
                                                                self.T,
                                                                show_bar=False)
        #baseline.fit(paths)
        sub_observations=[p["observations"] for p in sub_paths]
        sub_actions = [p["actions"] for p in sub_paths]
        sub_d_rewards = [p["rewards"] for p in sub_paths]

        temp = list()
        for x in sub_d_rewards:
            z=list()
            t=1
            for y in x:
                z.append(y*t)
                t*=self.discount
            temp.append(np.array(z))
        sub_d_rewards = temp

        iw = self.f_importance_weights(sub_observations[0],sub_actions[0])
        importance_weights = [np.mean(iw)]
        
        s_g = self.gradient_approx
        g = self.f_train_SVRG(sub_observations[0],sub_actions[0],sub_d_rewards[0],s_g[0],s_g[1],s_g[2],s_g[3],iw)
        for ob,ac,rw in zip(sub_observations[1:],sub_actions[1:],sub_d_rewards[1:]):
            iw = self.f_importance_weights(ob,ac)
            importance_weights.append(np.mean(iw))
            g = [sum(x) for x in zip(g,self.f_train_SVRG(ob,ac,rw,s_g[0],s_g[1],s_g[2],s_g[3],iw))]
        g = [x/len(sub_paths) for x in g]
        self.f_update(g[0],g[1],g[2],g[3])

        p=self.snap_policy.get_param_values(trainable=True)
        s_p = parallel_sampler.sample_paths_on_trajectories(self.policy.get_param_values(),
                                                            self.M,
                                                            self.T,
                                                            show_bar=False)
        self.snap_policy.set_param_values(p,trainable=True)
        
        rewards_sub = np.array([sum(p["rewards"]) for p in s_p])
        avg_return = np.mean([sum(p["rewards"]) for p in s_p])
        print('Average Return:', avg_return)


    def init_SVRG(self):
        parallel_sampler.populate_task(self.env, self.snap_policy)
        dist = self.policy.distribution
        snap_dist = self.snap_policy.distribution

        ''' Create Theano variables '''
        observations_var = self.env.observation_space.new_tensor_variable(
            'observations',
            # It should have 1 extra dimension since we want to represent a list of observations
            extra_dims=1
        )
        actions_var = self.env.action_space.new_tensor_variable(
            'actions',
            extra_dims=1
        )
        returns_var = TT.vector('returns')
        d_rewards_var = TT.vector('d_rewards')
        importance_weights_var = TT.vector('importance_weight')

        ''' Create Symbolic expressions '''
        dist_info_vars = self.policy.dist_info_sym(observations_var)
        snap_dist_info_vars = self.snap_policy.dist_info_sym(observations_var)
    
        # Negate the objective because SGD performes Stochastic Gradient Descent and we want a SG Assent
        surr = TT.sum(- dist.log_likelihood_sym_1traj_GPOMDP(actions_var, dist_info_vars) * d_rewards_var)

        params = self.policy.get_params(trainable=True)
        snap_params = self.snap_policy.get_params(trainable=True)

        importance_weights = dist.likelihood_ratio_sym_1traj_GPOMDP(actions_var,snap_dist_info_vars,dist_info_vars)
        grad = theano.grad(surr, params)

        eval_grad1 = TT.matrix('eval_grad0',dtype=grad[0].dtype)
        eval_grad2 = TT.vector('eval_grad1',dtype=grad[1].dtype)
        eval_grad3 = TT.col('eval_grad3',dtype=grad[2].dtype)
        eval_grad4 = TT.vector('eval_grad4',dtype=grad[3].dtype)

        surr_on1 = TT.sum(- dist.log_likelihood_sym_1traj_GPOMDP(actions_var,dist_info_vars)*d_rewards_var*importance_weights_var)
        surr_on2 = TT.sum(snap_dist.log_likelihood_sym_1traj_GPOMDP(actions_var,snap_dist_info_vars)*d_rewards_var)
        grad_SVRG =[sum(x) for x in zip([eval_grad1, eval_grad2, eval_grad3, eval_grad4], theano.grad(surr_on1,params),theano.grad(surr_on2,snap_params))]

        ''' Create Symbolic functions '''
        self.f_train = theano.function(
            inputs = [observations_var, actions_var, d_rewards_var],
            outputs = grad
        )
        self.f_update = theano.function(
            inputs = grad,
            outputs = None,
            updates = sgd(grad, params, learning_rate=self.learning_rate)
        )
        self.f_importance_weights = theano.function(
            inputs = [observations_var, actions_var],
            outputs = importance_weights
        )
        self.f_train_SVRG = theano.function(
            inputs=[observations_var, actions_var, d_rewards_var, eval_grad1, eval_grad2, eval_grad3, eval_grad4,importance_weights_var],
            outputs=grad_SVRG,
        )