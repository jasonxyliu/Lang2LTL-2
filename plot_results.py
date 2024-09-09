import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# -- values for full system accuracy plot:
methods = ['Text+Image (Ours)', 'Text Only (Lang2LTL)', 'Image Only']
accuracy = [93.53, 82.81, 41.69]
std = [4.33, 9.54, 14.65]

# -- values for SRER/acc plot:
bins_props = ['1', '2', '3', '4', '5']
acc_props = [0.9969465648854962, 0.9947887323943662, 0.9967980295566502, 0.99321608040201, 0.9920398009950249]

# -- values for REG/acc plot:
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

plot_type = ['full_sys', 'reg_acc', 'spg_acc']

for P in plot_type:
	plt.figure(figsize=(6,3))
	if P == 'full_sys':
		plt.bar(x=methods, height=accuracy, color=sns.color_palette('colorblind'))
		plt.errorbar(methods, accuracy, yerr=std, color="k", fmt='.', elinewidth=2,capthick=2, ms=10, capsize=4)
		plt.title("Full System Accuracy", fontsize=18)
	elif P == 'reg_acc':
		plt.bar(x=bins_reg, height=acc_reg, color=sns.color_palette('colorblind')[0])
		plt.xlabel("Length of Referring Expressions (REs)", fontsize=16)
	else:
		plt.bar(x=bins_props, height=acc_props, color=sns.color_palette('colorblind')[0])
		plt.xlabel("Number of Spatial Referring Expressions (SREs)", fontsize=16)

	plt.ylabel("Accuracy (%)", fontsize=14)
	plt.show()
#endfor
