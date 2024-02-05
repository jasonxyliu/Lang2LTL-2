import os
import json
import dill
import csv
import string
import re

from random import randint, choice
from pandas import read_csv
from tqdm import tqdm
from time import sleep

def deserialize_props_str(props_str):
    """
    Deserialize json string of propositions.
    :param props_str: "('a',)", "('a', 'b')", "['a',]", "['a', 'b']",
    :return: ['a'], ['a', 'b'], ['a'], ['a', 'b']
    """
    props = [prop.translate(str.maketrans('', '', string.punctuation)).strip() for prop in list(props_str.strip("()[]").split(", "))]
    return props


def load_from_file(fpath, noheader=True):
    ftype = os.path.splitext(fpath)[-1][1:]
    if ftype == 'pkl':
        with open(fpath, 'rb') as rfile:
            out = dill.load(rfile)
    elif ftype == 'txt':
        with open(fpath, 'r') as rfile:
            if 'prompt' in fpath:
                out = "".join(rfile.readlines())
            else:
                out = [line.strip() for line in rfile.read().splitlines() if line]
    elif ftype == 'json':
        with open(fpath, 'r') as rfile:
            out = json.load(rfile)
    elif ftype == 'csv':
        with open(fpath, 'r') as rfile:
            csvreader = csv.reader(rfile)
            if noheader:
                fileds = next(csvreader)
            out = [row for row in csvreader]
    else:
        raise ValueError(f"ERROR: file type {ftype} not recognized")
    return out


def save_to_file(data, fpth, mode=None):
    ftype = os.path.splitext(fpth)[-1][1:]
    if ftype == 'pkl':
        with open(fpth, mode if mode else 'wb') as wfile:
            dill.dump(data, wfile)
    elif ftype == 'txt':
        with open(fpth, mode if mode else 'w') as wfile:
            wfile.write(data)
    elif ftype == 'json':
        with open(fpth, mode if mode else 'w') as wfile:
            json.dump(data, wfile, indent=4)
    elif ftype == 'csv':
        with open(fpth, mode if mode else 'w', newline='') as wfile:
            writer = csv.writer(wfile)
            writer.writerows(data)
    else:
        raise ValueError(f"ERROR: file type {ftype} not recognized")


def generate_dataset(params, utts_fpath, gtr_fpath, num_utterances=10, min_props=1, max_props=5):

    def load_ltl_samples(fpath):
        return read_csv(open(fpath, 'r'))

    def load_groundtruth(fpath):
        return load_from_file(fpath)

    # NOTE: we will pass params to this function:
    location = params['location']
    ltl_samples = load_ltl_samples(params['ltl_samples'])
    gtr = load_groundtruth(params['gtr'])[location]

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
        list_true_srer = []
        list_true_reg_spg = []

        existing_targets = []
        for P in range(len(ltl_props)):

            new_sre = None
            new_srer = {}

            while not bool(new_sre):
                random_pred = choice(gtr[choice(landmarks)])
                # -- this is an element without any spatial relation:

                target = None

                if "*" in random_pred:
                    new_sre = random_pred["*"]
                    new_srer = {new_sre : {}}

                    possible_targets = []
                    for x in landmarks:
                        for y in gtr[x]:
                            if '*' in y and y["*"] == new_sre:
                                possible_targets.append(x)

                    target = choice(possible_targets)

                    random_pred = {new_sre: target}
                else:
                    # -- this means we are using one of the groundtruth entries that have specific object instances:

                    # -- get the relation in this predicate:
                    rel = list(random_pred.keys())[0]

                    if len(random_pred[rel]) == 2 or len(random_pred[rel]) == 3:
                        target = random_pred[rel][0]

                        # -- we will do some "lifting" of the specific landmarks assigned to this SRE:
                        lifted_target = choice([x['@'] for x in gtr[random_pred[rel][0]] if '@' in x] +
                                                [x['*'] for x in gtr[random_pred[rel][0]] if '*' in x])

                        if len(random_pred[rel]) == 2:
                            try:
                                lifted_anchor = choice([x['*'] for x in gtr[random_pred[rel][1]] if '*' in x])
                            except IndexError:
                                continue

                            new_sre = f'{lifted_target} {rel} {lifted_anchor}'
                            new_srer[rel] = [lifted_target, lifted_anchor]

                        elif len(random_pred[rel]) == 3:
                            try:
                                lifted_anchor_1 = choice([x['*'] for x in gtr[random_pred[rel][1]] if '*' in x])
                            except IndexError:
                                continue

                            try:
                                lifted_anchor_2 = choice([x['*'] for x in gtr[random_pred[rel][2]] if '*' in x])
                            except IndexError:
                                continue

                            new_sre = f'{lifted_target} {rel} {lifted_anchor_1} and {lifted_anchor_2}'
                            new_srer[rel] = [lifted_target, lifted_anchor_1, lifted_anchor_2]

                if target in existing_targets:
                    new_sre = None

                # NOTE: we will keep track of all targets we have already added to make sure we don't get repeats:
                existing_targets.append(target)

            list_true_sre.append(new_sre)
            list_true_srer.append(new_srer)
            list_true_reg_spg.append(random_pred)

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

