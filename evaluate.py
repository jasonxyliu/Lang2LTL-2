from utils import load_from_file


def evaluate_spg(spg_output_fpth, gtr_fpath, topk):
	# NOTE: this function will take in a JSON outputted by the SPG module and it will produce a report
	#		of how many matches it gets right.

	# -- we need to count the total number of SREs across all commands in the JSON file:
	total_sres = 0
	total_topk = {f'top-{x+1}': 0 for x in range(topk)}

	spg_output = load_from_file(spg_output_fpth)
	gtr_data = load_from_file(gtr_fpath)
	gtr_utt_to_results = {x['utt']: x for x in gtr_data}

	for result in spg_output:
		print(result['utt'])
		gtr = gtr_utt_to_results[result['utt']]
		total_sres += len(gtr['true_reg_spg'])

		spgs_gtr = gtr['true_reg_spg']
		spgs_gen = result['spg_results']

		for spg_gtr in spgs_gtr:
			spg_gen = None
			for GR in spgs_gen:
				# for each groundtruth SPG result, we check if there is a computed result for it as well:
				if list(spg_gtr)[0] == list(GR)[0]:
					spg_gen = GR

			if not spg_gen:
				continue

			current_sre = list(spg_gtr)[0]

			print(current_sre)

			for k in range(topk):
				if spg_gen[current_sre][k]['target'] in spg_gtr[current_sre]:
					print(spg_gen[current_sre][k]['target'])
					print(spg_gtr[current_sre])
					print(k)
					for j in range(k, topk):
						total_topk[f'top-{j+1}'] += 1
					break

		print(len(gtr['true_reg_spg']))
		print(total_topk)
		input()

	print(total_sres)
	print(total_topk)
