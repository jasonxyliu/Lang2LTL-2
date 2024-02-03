import os
import json
import dill
import csv
import string
import re

from random import randint, choice
from pandas import read_csv

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


def generate_dataset(params, fpath, num_utterances=10, min_props=2):

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
    used_templates = []

    count = 0

    while count < num_utterances:
        # -- use the eval function to get the list of props for a randomly selected row from the set of LTL blueprints:
        ltl_sample = ltl_samples.iloc[randint(0, len(ltl_samples)-1)]
        ltl_props = eval(ltl_sample['props'])

        if len(ltl_props) < min_props:
            continue

        count += 1

        ltl_blueprint = ltl_sample['utterance']

        # -- save all original templates for matching them to reverse-engineered result later:
        used_templates.append(str(ltl_blueprint))

        if not ltl_blueprint.endswith('.'):
            ltl_blueprint += '.'

        # -- add a full-stop at the beginning and end of the sentence for easier tokenization:
        if not ltl_blueprint.startswith('.'):
            ltl_blueprint = '.' + ltl_blueprint

        new_command = ltl_blueprint
        for x in range(len(ltl_props)):

            target = choice(landmarks)

            random_pred = choice(gtr[target])

            # NOTE: to do replacement of the lifted proposition with the generated one, we need to account for
            # different ways it would be written preceded by a whitespace character, i.e., ' a ', ' a,', ' a.'
            new_command = re.sub(rf"(\W)([{ltl_props[x]}])(\W)", rf'\1{random_pred}\3', new_command)

            # NOTE: some utterances will be missing some propositions

        list_utterances.append(new_command[1:-1])

    utts_for_file = ""
    for command in list_utterances:
        utts_for_file = f'{command}\n{utts_for_file}'

    save_to_file(utts_for_file, fpath)
