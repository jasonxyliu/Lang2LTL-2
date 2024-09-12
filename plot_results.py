import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Avoid Type 3 fonts
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def compute_mean_err_bar(data, confidence=None):
	# Sample data
	data = np.array(data)

	# Sample mean and standard error
	mean = np.mean(data)
	std_error = stats.sem(data)  # standard error of the mean

	if confidence:  # 95% confidence interval
		# Compute the 95% confidence interval; use t-distribution since sample size < 30
		n = len(data)
		t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)  # t-critical value for 95% CI

		margin_of_error = t_critical * std_error
		# confidence_interval = (mean - margin_of_error, mean + margin_of_error)
		err_bar = margin_of_error
	else:
		err_bar = std_error
	return mean, err_bar


def plot_full_sys_acc():
	# Full system accuracy plot
	methods = ['Text+Image (Ours)', 'Text Only (Lang2LTL-Spatial)', 'Image Only']

	accuracies_per_sys = [
		[
			0.9531680441, 0.9348025712, 0.9467401286, 0.943067034, 0.9880624426,
			0.8576675849, 0.8503213958, 0.8383838384, 0.8714416896, 0.8622589532,
			0.8842975207, 0.8778696051, 0.8907254362, 0.8760330579, 0.8466483012,
			0.7520661157, 0.7594123049, 0.7731864096, 0.8797061524, 0.867768595
		],
		[
			0.3140495868, 0.3158861341, 0.3149678604, 0.3149678604, 0.3140495868,
			0.2736455464, 0.2837465565, 0.2837465565, 0.2883379247, 0.2883379247,
			0.2837465565, 0.277318641, 0.277318641, 0.2819100092, 0.2883379247,
			0.2736455464, 0.2883379247, 0.2883379247, 0.2764003673, 0.2653810836
		],
		[
			0.4756657484, 0.4710743802, 0.4775022957, 0.4729109275, 0.4710743802,
			0.3232323232, 0.3112947658, 0.290174472, 0.3269054178, 0.3269054178,
			0.6703397612, 0.7033976125, 0.6914600551, 0.665748393, 0.6831955923,
			0.5610651974, 0.5665748393, 0.5573921028, 0.5564738292, 0.5656565657
		]
	]

	means, err_bars = [], []
	for accuracies in accuracies_per_sys:
		# mean, err_bar = compute_mean_err_bar(accuracies)
		mean, err_bar = compute_mean_err_bar(accuracies, 0.95)
		means.append(mean)
		err_bars.append(err_bar)

	fig = plt.figure(figsize=(6,5))
	plt.bar(x=methods, height=means, color=sns.color_palette('colorblind'))
	plt.errorbar(methods, means, yerr=err_bars, color="k", fmt='.', elinewidth=2,capthick=2, ms=10, capsize=4)
	plt.title("Full System Accuracy", fontsize=18)
	plt.xlabel("Modality", fontsize=16)
	plt.ylabel("Accuracy (%)", fontsize=16)
	plt.xticks(fontsize=9)
	fig.tight_layout()
	fig.savefig("full_acc.pdf")


def plot_srer_acc():
	# SRER accuracy plot
	bins_props = ['1', '2', '3', '4', '5']
	acc_props = [0.9969465648854962, 0.9947887323943662, 0.9967980295566502, 0.99321608040201, 0.9920398009950249]

	fig = plt.figure(figsize=(6,3))
	plt.bar(x=bins_props, height=acc_props, color=sns.color_palette('colorblind')[0])
	plt.xlabel("Number of Spatial Referring Expressions (SREs)", fontsize=16)
	plt.ylabel("Accuracy (%)", fontsize=16)
	plt.xticks(fontsize=11)
	fig.tight_layout()
	fig.savefig("srer_acc.pdf")


def plot_reg_acc():
	# REG accuracy plot
	bins_reg = ['0-10', '11-20', '21-30', '31-40']
	acc_dict = {3: 1.0, 4: 1.0, 5: 0.9996715388405321, 6: 1.0, 7: 1.0, 8: 0.9831536388140162, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 0.9970760233918129, 21: 1.0, 22: 1.0, 23: 1.0, 24: 1.0, 25: 0.9983974358974359, 26: 1.0, 28: 1.0, 29: 1.0, 30: 1.0, 31: 1.0, 32: 1.0, 33: 1.0, 35: 1.0, 36: 1.0, 37: 0.9807692307692307}
	acc_reg = []
	for x in bins_reg:
		start, end = eval(x.split('-')[0]), eval(x.split('-')[1])
		total, count = 0, 0
		for y in range(start, end+1):
			if y in acc_dict:
				total += acc_dict[y]
				count += 1
		acc_reg.append(total/count)

	fig = plt.figure(figsize=(6,3))
	plt.bar(x=bins_reg, height=acc_reg, color=sns.color_palette('colorblind')[0])
	plt.xlabel("Length of Referring Expressions (REs)", fontsize=16)
	plt.ylabel("Accuracy (%)", fontsize=16)
	plt.xticks(fontsize=11)
	fig.tight_layout()
	fig.savefig("reg_acc.pdf")


if __name__ == "__main__":
	plot_full_sys_acc()
	plot_srer_acc()
	plot_reg_acc()
