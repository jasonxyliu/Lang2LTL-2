"""
Modular-wise Evaluation: correct input to each module, check modular output.
"""
import os
import argparse
import logging

from ground import LOC2GID
from srer import run_exp_srer
from reg import run_exp_reg
from spg import run_exp_spg
from lt import run_exp_lt
from lt_rag import run_exp_lt_rag
from evaluate import eval_srer, eval_reg, eval_spg, eval_lt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, default="all", choices=["srer", "reg", "spg", "lt", "all"], help="domain name.")
    parser.add_argument("--loc", type=str, default="boston", choices=["blackstone", "boston", "auckland", "san_francisco"], help="domain name.")
    parser.add_argument("--ablate", type=str, default=None, choices=["text", "image", None], help="ablate out a modality or None to use both")
    parser.add_argument("--nsamples", type=int, default=None, help="number of sample utts per LTL formula or None for all.")
    parser.add_argument("--seed", type=int, default=0, help="seed to random sampler.")  # 0, 1, 2, 42, 111
    parser.add_argument("--topk", type=int, default=10, help="top k most likely landmarks grounded by REG.")
    parser.add_argument("--lt", type=str, default="t5", choices=["t5", "rag"], help="lifted translation model.")
    parser.add_argument("--nexamples", type=int, default=5, help="number of in-context examples if use RAG lifted translation model.")
    args = parser.parse_args()
    loc_id = f"{args.loc}_n{args.nsamples}_seed{args.seed}" if args.nsamples else f"{args.loc}_all_seed{args.seed}"
    lt_id = f"lt-{args.lt}{args.nexamples}" if args.lt == "rag" else f"{args.lt}"

    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    graph_dpath = os.path.join(data_dpath, "maps", LOC2GID[args.loc])
    osm_fpath = os.path.join(data_dpath, "osm", f"{args.loc}.json")
    utts_fpath = os.path.join(data_dpath, "dataset", args.loc, f"{loc_id}_utts.txt")
    model_fpath = os.path.join(os.path.expanduser("~"), "ground", "models", "checkpoint-best")
    results_dpath = os.path.join(os.path.expanduser("~"), "ground", "results_modular", loc_id)
    os.makedirs(results_dpath, exist_ok=True)
    rel_embeds_fpath = os.path.join(results_dpath, f"known_rel_embeds.json")
    srer_out_fname = f"srer_outs_ablate_{args.ablate}.json" if args.ablate else f"srer_outs.json"
    srer_out_fpath = os.path.join(results_dpath, srer_out_fname)
    reg_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "reg"))
    reg_in_cache_fpath = os.path.join(results_dpath, "reg_in_cache.pkl")
    spg_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "spg"))
    lt_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", f"lt-{lt_id}"))
    true_results_fpath = os.path.join(data_dpath, "dataset", args.loc, f"{loc_id}_true_results.json")
    ltl_fpath = os.path.join(data_dpath, "dataset", "ltl_samples_sorted.csv")

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(results_dpath, f"eval_results_{args.module}.log"), mode='w'),
                            logging.StreamHandler()
                        ]
    )
    logging.info(f"***** Modular-wise Evaluation on Dataset: {loc_id}")

    if args.module == "srer" or args.module == "all":
        run_exp_srer(utts_fpath, srer_out_fpath)
        eval_srer(true_results_fpath, srer_out_fpath)

    if args.module == "reg" or args.module == "all":
        run_exp_reg(true_results_fpath, graph_dpath, osm_fpath, args.topk, args.ablate, reg_out_fpath, reg_in_cache_fpath)
        eval_reg(true_results_fpath, args.topk, reg_out_fpath)

    if args.module == "spg" or args.module == "all":
        run_exp_spg(true_results_fpath, graph_dpath, osm_fpath, args.topk, rel_embeds_fpath, spg_out_fpath)
        eval_spg(true_results_fpath, args.topk, spg_out_fpath)

    if args.module == "lt" or args.module == "all":
        if args.lt == "t5":
            run_exp_lt(true_results_fpath, model_fpath, lt_out_fpath)
        elif args.lt == "rag":
            run_exp_lt_rag(true_results_fpath, lt_out_fpath, data_dpath, ltl_fpath, args.nexamples)

        eval_lt(true_results_fpath, lt_out_fpath)
