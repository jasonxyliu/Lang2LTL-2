import os
import argparse
import re
from time import sleep
from random import randint, choice
from tqdm import tqdm


from utils import load_from_file, save_to_file


def generate_dataset(params, utts_fpath, gtr_fpath, num_utterances=10, min_props=1, max_props=5):
    # NOTE: we will pass params to this function:
    location = params['location']
    ltl_samples = load_from_file(params['ltl_samples'], use_pandas=True)
    gtr = load_from_file(params['gtr'])[location]

    # -- getting all landmarks or points of interest:
    landmarks = list(gtr.keys())
    list_utterances = []
    list_true_results = []
    count = 0

    progress_bar = tqdm(total=num_utterances, desc="Generating synthetic dataset and groundtruth file...")

    while count < num_utterances:
        # -- use the eval function to get the list of props for a randomly selected row from the set of LTL blueprints:
        ltl_sample = ltl_samples.iloc[randint(0, len(ltl_samples)-1)]
        ltl_props = list(set(eval(ltl_sample['props'])))

        if len(ltl_props) < min_props or len(ltl_props) > max_props:
            continue

        ltl_blueprint = ltl_sample['utterance']
        ltl_formula = ltl_sample['ltl_formula']

        if not ltl_blueprint.endswith('.'):
            ltl_blueprint += '.'

        # -- add a full-stop at the beginning and end of the sentence for easier tokenization:
        if not ltl_blueprint.startswith('.'):
            ltl_blueprint = '.' + ltl_blueprint

        new_command = ltl_blueprint

        list_true_sre = []
        list_true_srer = {}
        list_true_reg_spg = {}

        existing_targets = []
        for P in range(len(ltl_props)):

            new_sre = None
            new_srer = {}

            while not bool(new_sre):
                random_lmrk = choice(landmarks)
                random_pred = choice(gtr[random_lmrk])
                # -- this is an element without any spatial relation:

                target = None

                if "*" in random_pred or "@" in random_pred:
                    new_sre = random_pred["*" if "*" in random_pred else "@"]
                    new_srer = {new_sre : {}}
                    target = [random_lmrk]
                else:
                    # -- this means we are using one of the groundtruth entries that have specific object instances:

                    # -- get the relation in this predicate:
                    rel = list(random_pred.keys())[0]

                    if len(random_pred[rel]) == 2 or len(random_pred[rel]) == 3:
                        target = random_pred[rel][0]

                        # -- we will do some "lifting" of the specific landmarks assigned to this SRE:
                        lifted_target = choice([x['@'] for x in gtr[target[0]] if '@' in x] +
                                               [x['*'] for x in gtr[target[0]] if '*' in x])

                        if len(random_pred[rel]) == 2:
                            anchor = random_pred[rel][1]
                            lifted_anchor = choice([x['*'] for x in gtr[anchor[0]] if '*' in x])

                            new_sre = f'{lifted_target} {rel} {lifted_anchor}'
                            new_srer[rel] = [lifted_target, lifted_anchor]

                        elif len(random_pred[rel]) == 3:
                            anchor_1 = random_pred[rel][1]
                            anchor_2 = random_pred[rel][2]

                            lifted_anchor_1 = choice([x['*'] for x in gtr[anchor_1[0]] if '*' in x])
                            lifted_anchor_2 = choice([x['*'] for x in gtr[anchor_2[0]] if '*' in x])

                            new_sre = f'{lifted_target} {rel} {lifted_anchor_1} and {lifted_anchor_2}'
                            new_srer[rel] = [lifted_target, lifted_anchor_1, lifted_anchor_2]

                if set(existing_targets) & set(target):
                    new_sre = None
                    continue

                # NOTE: we will keep track of all targets we have already added to make sure we don't get repeats:
                existing_targets += target

                list_true_reg_spg[new_sre] = target
                list_true_srer[new_sre] = new_srer[list(new_srer)[0]]

            list_true_sre.append(new_sre)

            # NOTE: to do replacement of the lifted proposition with the generated one, we need to account for
            # different ways it would be written preceded by a whitespace character, i.e., ' a ', ' a,', ' a.'
            new_command = re.sub(rf"(\b)([{ltl_props[P]}])(\W)", rf'\1{new_sre}\3', new_command)

            # NOTE: some utterances will be missing some propositions

        list_utterances.append(new_command[1:-1])

        list_true_results.append({
            "utt": list_utterances[-1],
            "true_ltl": ltl_formula,
            "true_sre": list_true_sre,
            "true_srer": list_true_srer,
            "true_reg_spg": list_true_reg_spg
        })

        count += 1

        sleep(0.001)
        progress_bar.update(1)

    progress_bar.close()
    print()

    utts_for_file = ""
    for command in list_utterances:
        utts_for_file = f'{command}\n{utts_for_file}'

    save_to_file(utts_for_file, utts_fpath)
    save_to_file(list_true_results, gtr_fpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default="boston", choices=["indoor_env_0", "alley", "blackstone", "boston", "auckland"], help="domain name.")
    args = parser.parse_args()

    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    utt_fpath = os.path.join(data_dpath, f"utts_{args.location}.txt")

    if not os.path.isfile(utt_fpath):
        generate_dataset(
            params={
                "location": args.location,
                "gtr": os.path.join(data_dpath, "true_lmk_grounds.json"),
                "ltl_samples": os.path.join(data_dpath, "symbolic_batch12_noperm.csv")
            },
            utts_fpath=utt_fpath,
            gtr_fpath=os.path.join(data_dpath, f"true_results_{args.location}.json")
        )
