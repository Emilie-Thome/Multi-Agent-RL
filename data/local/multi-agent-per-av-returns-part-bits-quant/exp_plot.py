import csv
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# quantization_tuning = 1
# dir_prefix = 'multi_agent_per_av_returns_part_bits_quant_2021_04_08_15_53_34_000'
# participation_rates = [0.1, 0.5, 1]
# agents_number = 10
# average_periods = [1, 5, 10]

# quantization_tuning = 4
# dir_prefix = 'multi_agent_per_av_returns_part_bits_quant_2021_04_09_14_16_51_000'
# participation_rates = [0.3, 0.5, 0.6]
# agents_number = 10
# average_periods = [1, 5]

# quantization_tuning = 10
# dir_prefix = 'multi_agent_per_av_returns_part_bits_quant_2021_04_09_14_32_29_000'
# participation_rates = [0.2, 0.5, 0.8]
# agents_number = 10
# average_periods = [5]

# files_id = [str(i) for i in range(1,5)]
# file = '/progress.csv'

# nb_axs = int(np.ceil(len(average_periods)/2))
# fig = plt.figure(figsize=(10,5), dpi=100)
# axs = []

# for k_period, average_period in enumerate(average_periods):
# 	ax = fig.add_subplot(nb_axs, 2, k_period+1)
# 	axs.append(ax)
# 	for k_rate, participation_rate in enumerate(participation_rates):
# 		id_file = files_id[k_rate*len(average_periods) + k_period]
# 		transf_bits = [0]
# 		average_returns = [0]
# 		with open(dir_prefix + id_file + file) as csvfile:
# 			reader = csv.DictReader(csvfile)
# 			for row in reader:
# 				transf_bits.append(int(row['TransfBits']))
# 				average_returns.append(float(row['TotalAverageReturn']))
# 		ax.plot(transf_bits, average_returns, label='Participation rate among agents: ' + str(participation_rate))
# 	ax.set_title('Average period: ' + str(average_period))
# 	ax.set(xlabel='Transferred Bits', ylabel='Average returns')

# handles, labels = axs[0].get_legend_handles_labels()
# fig.legend(handles, labels, shadow=True, fancybox=True, bbox_to_anchor=[0.9, 0.3/nb_axs])
# fig.suptitle(dir_prefix + ' --- Number of agents: ' + str(agents_number))
# fig.savefig(dir_prefix)

'''
Multiple quantization_tunings
'''
# dir_prefix = 'multi_agent_per_av_returns_part_bits_quant_2021_04_09_14_43_48_000'
# quantization_tunings = [1, 5, 10, 15]
# participation_rate = 0.8
# agents_number = 10
# average_period = 5

dir_prefix = 'multi_agent_per_av_returns_part_bits_quant_2021_04_09_15_06_43_000'
quantization_tunings = [10, 20, 30, 40]
participation_rate = 0.8
agents_number = 10
average_period = 5

files_id = [str(i) for i in range(1,5)]
file = '/progress.csv'

fig = plt.figure(figsize=(10,5), dpi=100)
ax = fig.add_subplot()
for k_quant, quantization_tuning in enumerate(quantization_tunings):
	id_file = files_id[k_quant]
	transf_bits = [0]
	average_returns = [0]
	with open(dir_prefix + id_file + file) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			transf_bits.append(int(row['TransfBits']))
			average_returns.append(float(row['TotalAverageReturn']))
	ax.plot(transf_bits, average_returns, label='Quantization tuning: ' + str(quantization_tuning))
ax.set_title('Average period: ' + str(average_period) + ' ; Agents number: ' + str(agents_number) + ' ; Participation rate: ' + str(participation_rate))
ax.set(xlabel='Transferred Bits', ylabel='Average returns')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, shadow=True, fancybox=True, bbox_to_anchor=[0.9, 0.5])
fig.suptitle(dir_prefix)
fig.savefig(dir_prefix)