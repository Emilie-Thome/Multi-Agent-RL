import csv
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


dir_prefix = 'test_transfer_params_not_quant_2021_04_14_18_07_39_000'
quantization_tunings = [1, 10]
participation_rate = 1
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
	ax.plot(transf_bits, average_returns, label='Experiment: ' + str(k_quant))
ax.set_title('Average period: ' + str(average_period) + ' ; Agents number: ' + str(agents_number) + ' ; Participation rate: ' + str(participation_rate))
ax.set(xlabel='Transferred Bits', ylabel='Average returns')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, shadow=True, fancybox=True, bbox_to_anchor=[0.9, 0.5])
fig.suptitle('Entire parameters, not quantized')
fig.savefig(dir_prefix)