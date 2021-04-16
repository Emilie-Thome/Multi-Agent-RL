import csv
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


dir_prefix = 'test_discounts_2021_04_16_10_06_18_00'
discounts = [0.5, 0.75, 0.99]
participation_rates = [0.5, 1]
agents_numbers = [5, 10]
average_periods = [1, 5, 10]

files_id = ['0'+str(i) for i in range(1,10)] + [str(i) for i in range(10,37)]
file = '/progress.csv'


for k_per, average_period in enumerate(average_periods):
	fig = plt.figure(figsize=(20,10), dpi=70)
	axs = fig.subplots(len(participation_rates), len(agents_numbers))

	for k_dis, discount in enumerate(discounts):
		for k_part, participation_rate in enumerate(participation_rates):
			for k_agent, agents_number in enumerate(agents_numbers):
			
				id_file = files_id[len(participation_rates)*len(agents_numbers)*len(average_periods)*k_dis
					+ len(agents_numbers)*len(average_periods)*k_part
					+ len(average_periods)*k_agent
					+k_per]
				transf_bits = [0]
				average_returns = [0]
				with open(dir_prefix + id_file + file) as csvfile:
					reader = csv.DictReader(csvfile)
					for row in reader:
						transf_bits.append(int(row['TransfBits']))
						average_returns.append(float(row['TotalAverageReturn']))
				axs[k_part][k_agent].plot(transf_bits, average_returns, label='Discount: ' + str(discount))
				axs[k_part][k_agent].set_title('Average period: ' + str(average_period) + ' ; Agents number: ' + str(agents_number) + ' ; Participation rate: ' + str(participation_rate))
				axs[k_part][k_agent].set(xlabel='Transferred Bits', ylabel='Average returns')

	handles, labels = axs[0,0].get_legend_handles_labels()
	fig.legend(handles, labels, shadow=True, fancybox=True, bbox_to_anchor=[0.9, 0.3])
	fig.suptitle('Difference parameters, every agent update their mean parameters reference, quantized, average period: ' + str(average_period))
	fig.savefig(dir_prefix + 'average_period' + str(average_period))