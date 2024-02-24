from lt_s2s_sup_tcd import Seq2Seq
from utils import load_from_file, save_to_file


def lt(spg_outs, model_fpath):
    lt_module = Seq2Seq(model_fpath, "t5-base")
    for spg_out in spg_outs:
        lifted_utt = spg_out["lifted_utt"]
        query = lifted_utt.translate(str.maketrans('', '', ',.'))
        lifted_ltl = lt_module.type_constrained_decode([query])[0]
        spg_out["lifted_ltl"] = lifted_ltl
        print(f"{lifted_utt}\n{lifted_ltl}\n")

def run_lt(spg_out_fpath, model_fpath, lt_out_fpath):
    spg_outs = load_from_file(spg_out_fpath)
    lt(spg_outs, model_fpath)
    save_to_file(spg_outs, lt_out_fpath)
