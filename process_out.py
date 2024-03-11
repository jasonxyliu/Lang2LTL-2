"""
Simplify system output.
Used to generate true labels for human data.
"""
import os
from collections import defaultdict

from utils import load_from_file, save_to_file


if __name__ == "__main__":
    loc = "boston"
    out_fpath = os.path.join(os.path.expanduser("~"), "ground", "results_human", loc, "lt_outs.json")
    true_fpath = os.path.join(os.path.expanduser("~"), "ground", "data", "human_data", f"{loc}_true_results.json")
    keys = ["utt", "lifted_utt", "sre_to_preds", "grounded_sre_to_preds", "grounded_sps", "lifted_ltl"]

    outs = load_from_file(out_fpath)
    outs_true = []

    for out in outs:
        out_true = {}
        for key, val in out.items():
            if key in keys:
                if key == "grounded_sre_to_preds":
                    grounded_sre_to_preds = defaultdict(lambda: defaultdict(list))
                    for sre, sp_grounds in val.items():
                        for rel, grounds in sp_grounds.items():
                            for ground in grounds:
                                grounded_sre_to_preds[sre][rel].append([[[1.0, out_ground] for _, out_ground in ground[:1]]])
                    out_true[key] = grounded_sre_to_preds
                elif key == "grounded_sps":
                    out_true[key] = {sre: grounds[:1] for sre, grounds in val.items()}
                else:
                    out_true[key] = val
        outs_true.append(out_true)

    save_to_file(outs_true, true_fpath)
