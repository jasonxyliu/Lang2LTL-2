import matplotlib.pyplot as plt
import seaborn as sns

# -- remove type 3 fonts:
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def plot_full_sys_acc():
	# Full system accuracy plot
	methods = ['Text+Image (Ours)', 'Text Only (Lang2LTL)', 'Image Only']
	accuracy = [93.53, 82.81, 41.69]
	std = [4.33, 9.54, 14.65]

	fig = plt.figure(figsize=(6,4))
	plt.bar(x=methods, height=accuracy, color=sns.color_palette('colorblind'))
	plt.errorbar(methods, accuracy, yerr=std, color="k", fmt='.', elinewidth=2,capthick=2, ms=10, capsize=4)
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
