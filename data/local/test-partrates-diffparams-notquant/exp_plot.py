import csv
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


# dir_prefix = 'test_partrates_diffparams_notquant_2021_04_15_08_16_53_000'
# dir_prefix = 'test_partrates_diffparams_notquant_2021_04_15_08_59_14_000'
# participation_rates = [0.2, 0.5, 0.9]
# agents_numbers = [1, 5, 10]
# average_period = 1

# dir_prefix = 'test_partrates_diffparams_notquant_2021_04_15_09_13_50_000'
# participation_rates = [0.5, 0.7, 1]
# agents_numbers = [1, 5, 10]
# average_period = 1


dir_prefix = 'test_partrates_diffparams_notquant_2021_04_15_09_33_36_00'
participation_rates = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
agents_numbers = [1, 5, 10]
average_period = 1

files_id = ['0'+str(i) for i in range(1,10)] + [str(i) for i in range(10,19)]
file = '/progress.csv'

fig = plt.figure(figsize=(15,5), dpi=100)
axs = fig.subplots(1, len(agents_numbers))
for k_agent, agents_number in enumerate(agents_numbers):
	for k_partrate, participation_rate in enumerate(participation_rates):
		id_file = files_id[len(agents_numbers)*k_partrate + k_agent]
		transf_bits = [0]
		average_returns = [0]
		with open(dir_prefix + id_file + file) as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				transf_bits.append(int(row['TransfBits']))
				average_returns.append(float(row['TotalAverageReturn']))
			print(transf_bits)
			print(average_returns)
			plt.show(block=False)
			axs[k_agent].plot(transf_bits, average_returns, label='Participation rate: ' + str(participation_rate))
	axs[k_agent].ticklabel_format(style='sci')
	axs[k_agent].set_title('Agents number: ' + str(agents_number))
	axs[k_agent].set(xlabel='Transferred Bits', ylabel='Average returns')

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, shadow=True, fancybox=True, bbox_to_anchor=[0.9, 0.7])
fig.suptitle('Difference parameters, not quantized, average period: ' + str(average_period))
fig.savefig(dir_prefix)
plt.show()
