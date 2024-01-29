import os

from lt_s2s_sup_tcd import Seq2Seq


def ground(lifted_utt, model_fpath):
    query = lifted_utt.translate(str.maketrans('', '', ',.'))
    lt_module = Seq2Seq(model_fpath, "t5-base")
    lifted_ltl = lt_module.type_constrained_decode([query])[0]
    return lifted_ltl


if __name__ == "__main__":
    model_fpath = os.path.join(os.path.expanduser("~"), "ground", "models", "checkpoint-best")

    lifted_utt = "go to a at most five times"
    lifted_ltl = ground(lifted_utt, model_fpath)

    print(lifted_ltl)
