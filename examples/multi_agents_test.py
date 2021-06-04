from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize
import numpy as np
import random
import theano
import theano.tensor as TT
from rllab.sampler import parallel_sampler
from lasagne.updates import sgd
import matplotlib.pyplot as plt
from rllab.envs.gym_env import GymEnv
import pandas as pd

# TODO: instead of the gradient, the agents normalize the gradient and use the normalize gradient to update.

def unpack(i_g):
	i_g_arr = [np.array(x) for x in i_g]
	res = i_g_arr[0].reshape(i_g_arr[0].shape[0]*i_g_arr[0].shape[1])
	res = np.concatenate((res,i_g_arr[1]))
	res = np.concatenate((res,i_g_arr[2][0]))
	res = np.concatenate((res,i_g_arr[3]))
	return res

nb_agents = 3 				# Number of agents
participation_rate = 1 		# Participation rate among agents to communicate with the server
s_tot = nb_agents*100000	# Total number of trajectories
N = 10000 					# N trajectories collected per iteration
M = 10 						# M secondary trajectories collected
T = 100 					# Trajectories will have at most 100 time steps
discount = 0.99 			# Discount factor for the objective
learning_rate = 0.0001 		# Learning rate for the gradient update
perc_est = 0.9 				# Perc estimate
partition = 3 # 3
porz = np.int(perc_est*N)

env = normalize(CartpoleEnv())

class Agent(object):

    def __init__(self):
    	self.env = env # same environment for every agent
    	self.policy = GaussianMLPPolicy(self.env.spec, hidden_sizes=(8,),learn_std=False)
    	self.snap_policy = GaussianMLPPolicy(self.env.spec, hidden_sizes=(8,),learn_std=False)
    	self.back_up_policy = GaussianMLPPolicy(self.env.spec, hidden_sizes=(8,),learn_std=False)
    	parallel_sampler.populate_task(self.env, self.snap_policy)
    	dist = self.policy.distribution
    	snap_dist = self.snap_policy.distribution
    	observations_var = self.env.observation_space.new_tensor_variable('observations', extra_dims=1)
    	actions_var = self.env.action_space.new_tensor_variable('actions', extra_dims=1)
    	d_rewards_var = TT.vector('d_rewards')
    	importance_weights_var = TT.vector('importance_weight')
    	dist_info_vars = self.policy.dist_info_sym(observations_var)
    	snap_dist_info_vars = self.snap_policy.dist_info_sym(observations_var)
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
    	grad_SVRG_4v = [sum(x) for x in zip(theano.grad(surr_on1,params),theano.grad(surr_on2,snap_params))]
    	grad_var = theano.grad(surr_on1,params)
    	self.f_train = theano.function(
    		inputs = [observations_var, actions_var, d_rewards_var],
    		outputs = grad
    		)
    	self.f_update = theano.function(
    		inputs = grad,
    		outputs = None,
    		updates = sgd(loss_or_grads=grad,
    			params=params,
    			learning_rate=learning_rate)
    		)
    	self.f_importance_weights = theano.function(
    		inputs = [observations_var, actions_var],
    		outputs = importance_weights
    		)
    	self.f_train_SVRG = theano.function(
    		inputs=[observations_var, actions_var, d_rewards_var, eval_grad1, eval_grad2, eval_grad3, eval_grad4,importance_weights_var],
    		outputs=grad_SVRG
    		)
    	self.f_train_SVRG_4v = theano.function(
    		inputs=[observations_var, actions_var, d_rewards_var,importance_weights_var],
    		outputs=grad_SVRG_4v
    		)
    	self.var_SVRG = theano.function(
    		inputs=[observations_var, actions_var, d_rewards_var, importance_weights_var],
    		outputs=grad_var
    		)

    def compute_snap_batch(self,observations,actions,d_rewards,n_traj,n_part):
    	n=n_traj
    	i=0
    	svrg_snap=list()
    	while(n-np.int(n_traj/n_part)>=0):
    		n=n-np.int(n_traj/n_part)
    		s_g = self.f_train(observations[i], actions[i], d_rewards[i])
    		for s in range(i+1,i+np.int(n_traj/n_part)):
    			s_g = [sum(x) for x in zip(s_g,self.f_train(observations[s], actions[s], d_rewards[s]))]
    		s_g = [x/np.int(n_traj/n_part) for x in s_g]
    		i += np.int(n_traj/n_part)
    		svrg_snap.append(unpack(s_g))
    	return svrg_snap

    def estimate_variance(self,observations,actions,d_rewards,snap_grads,n_traj,n_traj_s,n_part,M,N):
    	n=n_traj
    	i=0
    	svrg=list()
    	j=0
    	while(n-np.int(n_traj/n_part)>=0):
    		n=n-np.int(n_traj/n_part)
    		iw = self.f_importance_weights(observations[i],actions[i])
    		x = unpack(self.f_train_SVRG_4v(observations[i],actions[i],d_rewards[i],iw))*np.sqrt(np.int(n_traj/n_part)/M)
    		g = snap_grads[j]*np.sqrt(np.int(n_traj_s/n_part)/N)+x
    		for s in range(i+1,i+np.int(n_traj/n_part)):
    			iw = self.f_importance_weights(observations[s],actions[s])
    			g_prov=unpack(self.f_train_SVRG_4v(observations[s],actions[s],d_rewards[s],iw))*np.sqrt(np.int(n_traj/n_part)/M)
    			g+=snap_grads[j]*np.sqrt(np.int((n_traj_s)/n_part)/N) + g_prov
    		g=g/n_traj*n_part
    		i+=np.int(n_traj/n_part)
    		j+=1
    		svrg.append(g)
    	return (np.diag(np.cov(np.matrix(svrg),rowvar=False)).sum())


agents = [Agent() for i in range(nb_agents)]

variance_svrg_data={}
variance_sgd_data={}
importance_weights_data={}
rewards_snapshot_data={}
rewards_subiter_data={}
n_sub_iter_data={}

for k in range(1):
	avg_return = list()
	n_sub_iter=[]
	rewards_sub_iter=[]
	rewards_snapshot=[]
	importance_weights=[]
	variance_svrg = []
	variance_sgd = []

	#np.savetxt("policy_novar.txt",snap_policy.get_param_values(trainable=True))
	j=0
	while j<s_tot-N:
		observations = dict()
		actions = dict()
		d_rewards = dict()
		for agent in agents:
			j+=N
			paths = parallel_sampler.sample_paths_on_trajectories(agent.snap_policy.get_param_values(),N,T,show_bar=False)
			#baseline.fit(paths)
			observations[agent] = [p["observations"] for p in paths]
			actions[agent] = [p["actions"] for p in paths]
			d_rewards[agent] = [p["rewards"] for p in paths]
			temp = list()
			for x in d_rewards[agent]:
				z=list()
				t=1
				for y in x:
					z.append(y*t)
					t*=discount
				temp.append(np.array(z))
			d_rewards[agent]=temp
			s_g = agent.f_train(observations[agent][0], actions[agent][0], d_rewards[agent][0])
			s_g_fv = [unpack(s_g)]
			for ob,ac,rw in zip(observations[agent][1:],actions[agent][1:],d_rewards[agent][1:]):
				i_g = agent.f_train(ob, ac, rw)
				s_g_fv.append(unpack(i_g))
				s_g = [sum(x) for x in zip(s_g,i_g)]
			s_g = [x/len(paths) for x in s_g]
			
			b = agent.compute_snap_batch(observations[agent][0:porz],actions[agent][0:porz],d_rewards[agent][0:porz],porz,partition)
			agent.f_update(s_g[0],s_g[1],s_g[2],s_g[3])
			rewards_snapshot.append(np.array([sum(p["rewards"]) for p in paths])) 
			avg_return.append(np.mean([sum(p["rewards"]) for p in paths]))

			var_sgd = np.cov(np.matrix(b),rowvar=False)
			var_batch = (var_sgd)*(porz/partition)/M
			
			print(str(j-1)+' Average Return to compute Agent Gradient:', avg_return[-1])

		nb_participants = int(participation_rate*len(agents))
		participants = set()
		while len(participants) != nb_participants:
			participants.update({agents[random.randrange(len(agents))]})
		new_params = np.average([agent.policy.get_param_values(trainable=True) for agent in participants], axis=0)
		for agent in agents:
			agent.policy.set_param_values(new_params, trainable=True)
			agent.back_up_policy.set_param_values(new_params, trainable=True)

			n_sub = 0
			while j<s_tot-M:
				j += M
				n_sub+=1
				
				iw_var = agent.f_importance_weights(observations[agent][0],actions[agent][0])
				s_g_is = agent.var_SVRG(observations[agent][0], actions[agent][0], d_rewards[agent][0],iw_var)
				s_g_fv_is = [unpack(s_g_is)]
				for ob,ac,rw in zip(observations[agent][1:],actions[agent][1:],d_rewards[agent][1:]):
					iw_var = agent.f_importance_weights(ob, ac)
					s_g_is = agent.var_SVRG(ob, ac, rw,iw_var)
					s_g_fv_is.append(unpack(s_g_is))
				var_svrg = (agent.estimate_variance(observations[agent][porz:],actions[agent][porz:],d_rewards[agent][porz:],b,N-porz,porz,partition,M,N))
				var_dif = var_svrg-(np.diag(var_batch).sum())
				#eigval = np.real(np.linalg.eig(var_dif)[0])
				if (var_dif>0 or np.mean(iw_var)<0.6):
					print("1") 
					agent.policy.set_param_values(agent.back_up_policy.get_param_values(trainable=True), trainable=True)
					break
				variance_svrg.append(var_svrg)
				variance_sgd.append((np.diag(var_batch).sum()))

				#print(np.sum(eigval))
				
				
				sub_paths = parallel_sampler.sample_paths_on_trajectories(agent.snap_policy.get_param_values(),M,T,show_bar=False)
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
						t*=discount
					temp.append(np.array(z))
				sub_d_rewards = temp
				iw = agent.f_importance_weights(sub_observations[0],sub_actions[0])
				importance_weights.append(np.mean(iw))
				agent.back_up_policy.set_param_values(agent.policy.get_param_values(trainable=True), trainable=True) 
				
				g = agent.f_train_SVRG(sub_observations[0],sub_actions[0],sub_d_rewards[0],s_g[0],s_g[1],s_g[2],s_g[3],iw)
				for ob,ac,rw in zip(sub_observations[1:],sub_actions[1:],sub_d_rewards[1:]):
					iw = agent.f_importance_weights(ob,ac)
					importance_weights.append(np.mean(iw))
					g = [sum(x) for x in zip(g,agent.f_train_SVRG(ob,ac,rw,s_g[0],s_g[1],s_g[2],s_g[3],iw))]
				g = [x/len(sub_paths) for x in g]
				agent.f_update(g[0],g[1],g[2],g[3])

				p = agent.snap_policy.get_param_values(trainable=True)
				s_p = parallel_sampler.sample_paths_on_trajectories(agent.policy.get_param_values(),10,T,show_bar=False)
				agent.snap_policy.set_param_values(p,trainable=True)
				rewards_sub_iter.append(np.array([sum(p["rewards"]) for p in s_p]))
				avg_return.append(np.mean([sum(p["rewards"]) for p in s_p]))
				
				print(str(j)+' Average Return Agent solo:', avg_return[-1])
			print("2")
			n_sub_iter.append(n_sub)
			agent.snap_policy.set_param_values(agent.policy.get_param_values(trainable=True), trainable=True)

	rewards_subiter_data["rewardsSubIter"+str(k)]=rewards_sub_iter
	rewards_snapshot_data["rewardsSnapshot"+str(k)]= rewards_snapshot
	n_sub_iter_data["nSubIter"+str(k)]= n_sub_iter
	variance_sgd_data["variancceSgd"+str(k)] = variance_sgd
	variance_svrg_data["varianceSvrg"+str(k)]=variance_svrg
	importance_weights_data["importanceWeights"+str(k)] = importance_weights

	avg_return=np.array(avg_return)
	#plt.plot(avg_return)
	#plt.show()
rewards_subiter_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in rewards_subiter_data.items() ]))
rewards_snapshot_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in rewards_snapshot_data.items() ]))
n_sub_iter_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in n_sub_iter_data.items() ]))
variance_sgd_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in variance_sgd_data.items() ]))
variance_svrg_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in variance_svrg_data.items() ]))
importance_weights_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in importance_weights_data.items() ]))

rewards_subiter_data.to_csv("rewards_subiter.csv",index=False)
rewards_snapshot_data.to_csv("rewards_snapshot.csv",index=False)
n_sub_iter_data.to_csv("n_sub_iter.csv",index=False)
variance_sgd_data.to_csv("variance_sgd.csv",index=False)
variance_svrg_data.to_csv("variance_svrg.csv",index=False)
importance_weights_data.to_csv("importance_weights.csv",index=False)