import csv
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import ast

nb_agents = 10 			# Number of agents
participation_rate = 1 	# Participation rate among agents to communicate with the server
s_tot = nb_agents*10000 # Total number of trajectories
N = 100 				# N trajectories collected per iteration
T = 100 				# Trajectories will have at most 100 time steps
M = 10 					# M secondary trajectories collected
discount = 0.99 		# Discount factor for the objective
learning_rate = 0.0001 	# Learning rate for the gradient update
perc_est = 0.6 			# Perc estimate
partition = 3

files = {'importance_weights' : 'importance_weights.csv',
		'n_sub_iter' : 'n_sub_iter.csv',
		'rewards_snapshot' : 'rewards_snapshot.csv',
		'rewards_subiter' : 'rewards_subiter.csv',
		'variance_sgd' : 'variance_sgd.csv',
		'variance_svrg' : 'variance_svrg.csv'}

data = {'importance_weights' : [],
		'n_sub_iter' : [],
		'rewards_snapshot' : [],
		'rewards_subiter' : [],
		'variance_sgd' : [],
		'variance_svrg' : []}

# with open(files['rewards_snapshot']) as csvfile:
# 	reader = csv.reader(csvfile)
# 	first_line = True
# 	for row in reader:
# 		if first_line:
# 			print(row[0])
# 			first_line = False
# 		else:
# 			data['rewards_snapshot'].append([float(i.strip()) for i in row[0][1:-1].split(" ") if i.strip() != ''])

for key in files.keys():
	if key.find('rewards') != -1:
		with open(files[key]) as csvfile:
			reader = csv.reader(csvfile)
			first_line = True
			for row in reader:
				if first_line:
					print(row[0])
					first_line = False
				else:
					data[key].append(np.average([float(i.strip()) for i in row[0][1:-1].split(" ") if i.strip() != '']))
					# print(data[key])
					# break
	else:
		with open(files[key]) as csvfile:
			reader = csv.reader(csvfile)
			first_line = True
			for row in reader:
				if first_line:
					print(row[0])
					first_line = False
				else :
					data[key].append(ast.literal_eval(row[0]))
					# print(data[key])
					# break

rewards_subiter = data['rewards_subiter']
data['rewards_subiter'] = [[] for a in range(nb_agents)]
loc = 0
for i in range(len(data['n_sub_iter'])//nb_agents):
	for a in range(nb_agents):
		prev_loc = loc
		loc += data['n_sub_iter'][i*nb_agents+a]
		data['rewards_subiter'][a] += rewards_subiter[prev_loc:loc]


slice_by_agents = lambda l : [[l[i*nb_agents+a] for i in range(len(l)//nb_agents)] for a in range(nb_agents)]
data['n_sub_iter'] = slice_by_agents(data['n_sub_iter'])
data['rewards_snapshot'] = slice_by_agents(data['rewards_snapshot'])

for agent in range(nb_agents):
	fig = plt.figure(figsize=(10,5), dpi=200)
	ax = fig.add_subplot()
	
	rewards_snapshot = data['rewards_snapshot'][agent]
	data['rewards_snapshot'][agent] = []
	for ind, n_sub in enumerate(data['n_sub_iter'][agent]):
		data['rewards_snapshot'][agent] = data['rewards_snapshot'][agent] + [rewards_snapshot[ind]]*(n_sub-1)
	ax.plot(data['rewards_snapshot'][agent], label=str(agent+1)+'- Average of the N rewards')
	ax.plot(data['rewards_subiter'][agent], label=str(agent+1)+'- Average of the M rewards')

	ax.set_title('Agents number: ' + str(nb_agents) + ' ; Participation rate: ' + str(participation_rate))
	ax.set(xlabel='Analog to iterations', ylabel='Average rewards')

	handles, labels = ax.get_legend_handles_labels()
	fig.legend(handles, labels, bbox_to_anchor=[0.904, 0.89])
	fig.suptitle(' ')
	plt.show(block=False)
plt.show()