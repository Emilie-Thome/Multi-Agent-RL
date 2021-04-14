import csv
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

dir_prefix = 'multi_agent_per_av_part_bits_2021_04_08_10_46_06_000'

participation_rates = [0.1, 0.5, 1]
agents_number = 10
average_periods = [1, 5, 10]

files_id = [str(i) for i in range(1,10)]
file = '/progress.csv'

nb_axs = int(np.ceil(len(average_periods)/2))
fig = plt.figure()
axs = []

for k_period, average_period in enumerate(average_periods):
	ax = fig.add_subplot(nb_axs, 2, k_period+1)
	axs.append(ax)
	for k_rate, participation_rate in enumerate(participation_rates):
		id_file = files_id[k_rate*len(average_periods) + k_period]
		transf_bits = [0]
		average_returns = [0]
		with open(dir_prefix + id_file + file) as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				transf_bits.append(int(row['TransfBits']))
				average_returns.append(float(row['AverageReturn']))
		ax.plot(transf_bits, average_returns, label='Participation rate among agents: ' + str(participation_rate))
	ax.set_title('Average period: ' + str(average_period))
	ax.set(xlabel='Transferred Bits', ylabel='Average returns')

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, shadow=True, fancybox=True, bbox_to_anchor=[0.75, 0.75/nb_axs])
fig.suptitle('Number of agents: ' + str(agents_number))
plt.show()