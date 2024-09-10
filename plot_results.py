import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Avoid Type 3 fonts
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def compute_ci_interval(data, confience=0.95):
	# Sample data
	data = np.array(data)

	# Sample mean and standard error
	mean = np.mean(data)
	std_error = stats.sem(data)  # standard error of the mean

	# Compute the 95% confidence interval; use t-distribution since sample size < 30
	confidence = confience
	n = len(data)
	t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)  # t-critical value for 95% CI

	margin_of_error = t_critical * std_error
	# confidence_interval = (mean - margin_of_error, mean + margin_of_error)
	return mean, margin_of_error


def plot_full_sys_acc():
	# Full system accuracy plot
	methods = ['Text+Image (Ours)', 'Text Only (Lang2LTL)', 'Image Only']

	accuracies_per_sys = [
		[
			0.9917355372, 0.9908172635, 0.9898989899, 0.9917355372, 0.940312213,
			0.9632690542, 0.9678604224, 0.9595959596, 0.9834710744, 0.9550045914,
			0.9008264463, 0.8870523416, 0.8870523416, 0.8925619835, 0.870523416,
			0.9072543618, 0.9182736455, 0.9127640037, 0.927456382, 0.867768595
		],
		[
			0.8916437098,
			0.8668503214,
			0.6859504132,
			0.867768595
		],
		[
			0.4719926538,
			0.4380165289,
			0.2084481175,
			0.54912764
		]
	]

	means, margins = [], []
	for accuracies in accuracies_per_sys:
		mean, margin = compute_ci_interval(accuracies)
		means.append(mean)
		margins.append(margin)

	fig = plt.figure(figsize=(6,5))
	plt.bar(x=methods, height=means, color=sns.color_palette('colorblind'))
	plt.errorbar(methods, means, yerr=margins, color="k", fmt='.', elinewidth=2,capthick=2, ms=10, capsize=4)
	plt.title("Full System Accuracy", fontsize=18)
	plt.xlabel("Modality", fontsize=16)
	plt.ylabel("Accuracy (%)", fontsize=16)
	plt.xticks(fontsize=11)
	fig.tight_layout()
	fig.savefig("full_sys_acc.pdf")


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
	plt.xlabel("Length of Regular Expressions (REs)", fontsize=16)
	plt.ylabel("Accuracy (%)", fontsize=16)
	plt.xticks(fontsize=11)
	fig.tight_layout()
	fig.savefig("reg_acc.pdf")


if __name__ == "__main__":
	plot_full_sys_acc()
	plot_srer_acc()
	plot_reg_acc()
