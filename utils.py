import os
import json
import dill
import csv
import string
from pandas import read_csv


def deserialize_props_str(props_str):
    """
    Deserialize json string of propositions.
    :param props_str: "('a',)", "('a', 'b')", "['a',]", "['a', 'b']",
    :return: ['a'], ['a', 'b'], ['a'], ['a', 'b']
    """
    props = [prop.translate(str.maketrans('', '', string.punctuation)).strip() for prop in list(props_str.strip("()[]").split(", "))]
    return props


def load_from_file(fpath, noheader=True, use_pandas=False):
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
            if use_pandas:
                out = read_csv(rfile)
            else:
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


def copy_lt_outs(lt_out_fpath_from, lt_out_fpath_to, spg_out_fpath):
    """
    Optimization.
    When input utterance are the same, SRER and LT outputs are the same.
    """
    lt_outs = load_from_file(lt_out_fpath_from)
    spg_outs = load_from_file(spg_out_fpath)
    lt_outs_new = []

    for lt_out, spg_out in zip(lt_outs, spg_outs):
        assert lt_out["utt"].strip() == spg_out["utt"].strip(), f"ERROR different utterances:\ntrue: {lt_out['utt']}\npred: {spg_out['utt']}"

        spg_out["lifted_ltl"] = lt_out["lifted_ltl"]
        lt_outs_new.append(spg_out)

    save_to_file(lt_outs_new, lt_out_fpath_to)
