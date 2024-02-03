import os
import json
import dill
import csv
import string
import re

from random import randint, choice
from pandas import read_csv
from tqdm import tqdm

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


def generate_dataset(params, utts_fpath, gtr_fpath, num_utterances=10, min_props=2):

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

    for N in tqdm(range(num_utterances), desc="Generating synthetic dataset and groundtruth file..."):
        # -- use the eval function to get the list of props for a randomly selected row from the set of LTL blueprints:
        ltl_sample = ltl_samples.iloc[randint(0, len(ltl_samples)-1)]
        ltl_props = list(set(eval(ltl_sample['props'])))

        if len(ltl_props) < min_props:
            N -= 1
            continue

        ltl_blueprint = ltl_sample['utterance']
        ltl_formula = ltl_sample['ltl_formula']

        if not ltl_blueprint.endswith('.'):
            ltl_blueprint += '.'

        # -- add a full-stop at the beginning and end of the sentence for easier tokenization:
        if not ltl_blueprint.startswith('.'):
            ltl_blueprint = '.' + ltl_blueprint

        list_true_srer = []
        list_true_sre = []

        new_command = ltl_blueprint
        for x in range(len(ltl_props)):

            target = choice(landmarks)

            new_sre = None
            while not bool(new_sre):
                random_pred = choice(gtr[target])
                # -- this is an element without any spatial relation:
                if "*" in random_pred:
                    new_sre = random_pred["*"]
                    random_pred = {new_sre : {}}
                else:
                    rel = list(random_pred.keys())[0]
                    if len(random_pred[rel]) == 2:
                        new_sre = f'{random_pred[rel][0]} {rel} {random_pred[rel][1]}'
                    elif len(random_pred[rel]) == 3:
                        new_sre = f'{random_pred[rel][0]} {rel} {random_pred[rel][1]} and {random_pred[rel][2]}'

            list_true_sre.append(new_sre)
            list_true_srer.append(random_pred)

            # NOTE: to do replacement of the lifted proposition with the generated one, we need to account for
            # different ways it would be written preceded by a whitespace character, i.e., ' a ', ' a,', ' a.'
            new_command = re.sub(rf"(\W)([{ltl_props[x]}])(\W)", rf'\1{new_sre}\3', new_command)

            # NOTE: some utterances will be missing some propositions

        list_utterances.append(new_command[1:-1])

        list_true_results.append({
            "utt": list_utterances[-1],
            "true_ltl": ltl_formula,
            "true_srer": list_true_srer,
            "true_sre": list_true_sre
        })

    utts_for_file = ""
    for command in list_utterances:
        utts_for_file = f'{command}\n{utts_for_file}'

    save_to_file(utts_for_file, utts_fpath)
    save_to_file(list_true_results, gtr_fpath)

    print()
