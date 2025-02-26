import os
from tqdm import tqdm
import string

from lt_s2s_sup_tcd import Seq2Seq
from utils import load_from_file, save_to_file


def lt(spg_out, lt_model):
    lifted_utt = spg_out["lifted_utt"]
    query = lifted_utt.translate(str.maketrans('', '', string.punctuation))
    lifted_ltl = lt_model.type_constrained_decode([query])[0]
    spg_out["lifted_ltl"] = lifted_ltl


def run_exp_lt(spg_out_fpath, model_fpath, lt_out_fpath):
    if not os.path.isfile(lt_out_fpath):
        spg_outs = load_from_file(spg_out_fpath)
        lt_model = Seq2Seq(model_fpath, "t5-base")
        for spg_out in tqdm(spg_outs, desc="Running lifted translation (LT) module (method='t5-base')"):
            lt(spg_out, lt_model)
        save_to_file(spg_outs, lt_out_fpath)


if __name__ == "__main__":
    spg_out = {
        "lifted_utt": "you must avoid a only after you go to b"
        # "lifted_utt": "go to all of the following: a, b, and c"
    }
    model_fpath = os.path.join(os.path.expanduser("~"), "ground", "models", "checkpoint-best")
    lt_model = Seq2Seq(model_fpath, "t5-base")
    lt(spg_out, lt_model)

    print(f"Utt: {spg_out['lifted_utt']}\nLTL: {spg_out['lifted_ltl']}")
    breakpoint()
