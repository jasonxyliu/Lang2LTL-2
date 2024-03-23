"""
Modular-wise Evaluation: correct input to each module, check modular output.
"""
import os
import argparse
import logging
from shutil import copy2

from srer import run_exp_srer
from reg import run_exp_reg
from spg import run_exp_spg
from lt import run_exp_lt
from lt_rag import run_exp_lt_rag
from evaluate import eval_srer, eval_reg, eval_spg, eval_lt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, default="all", choices=["srer", "reg", "spg", "lt", "all"], help="domain name.")
    parser.add_argument("--loc", type=str, default="providence", choices=["providence", "auckland", "boston", "san_francisco"], help="domain name.")
    parser.add_argument("--ablate", type=str, default="both", choices=["both", "image", "text", None], help="ablate out a modality.")
    parser.add_argument("--nsamples", type=int, default=None, help="number of sample utts per LTL formula or None for all.")
    parser.add_argument("--seed", type=int, default=111, help="seed to random sampler.")  # 0, 1, 2, 42, 111 (resreved for ablate)
    parser.add_argument("--topk", type=int, default=10, help="top k most likely landmarks grounded by REG.")
    parser.add_argument("--lt", type=str, default="t5", choices=["t5", "rag"], help="lifted translation model.")
    parser.add_argument("--nexamples", type=int, default=2, help="number of in-context examples if use RAG lifted translation model.")
    args = parser.parse_args()
    loc_id = f"{args.loc}_n{args.nsamples}_seed{args.seed}" if args.nsamples else f"{args.loc}_all_seed{args.seed}"
    lt_id = f"lt-{args.lt}{args.nexamples}" if args.lt == "rag" else f"{args.lt}"

    data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
    graph_dpath = os.path.join(data_dpath, "maps", f"{args.loc}_ablate" if args.ablate else args.loc)
    osm_fpath = os.path.join(data_dpath, "osm_ablate" if args.ablate else "osm", f"{args.loc}.json")
    utts_fpath = os.path.join(data_dpath, "dataset", f"{args.loc}_ablate" if args.ablate else f"{args.loc}", f"{loc_id}_utts.txt")
    model_fpath = os.path.join(os.path.expanduser("~"), "ground", "models", "checkpoint-best")
    rel_embeds_fpath = os.path.join(data_dpath, f"known_rel_embeds.json")
    reg_in_cache_fpath = os.path.join(data_dpath, f"reg_in_cache_{args.loc}.pkl")
    results_dpath = os.path.join(os.path.expanduser("~"), "ground", f"results_modular_ablate_{args.ablate}" if args.ablate else "results_modular", loc_id)
    os.makedirs(results_dpath, exist_ok=True)
    srer_out_fname = "srer_outs.json"
    srer_out_fpath = os.path.join(results_dpath, srer_out_fname)
    reg_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "reg"))
    spg_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "spg"))
    lt_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", f"lt-{lt_id}"))
    true_results_fpath = os.path.join(data_dpath, "dataset",f"{args.loc}_ablate" if args.ablate else f"{args.loc}", f"{loc_id}_true_results.json")
    ltl_fpath = os.path.join(data_dpath, "dataset", "ltl_samples_sorted.csv")

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(results_dpath, f"eval_results_modular_{args.module}.log"), mode='w'),
                            logging.StreamHandler()
                        ]
    )
    logging.info(f"***** Modular-wise Evaluation Ablate {args.ablate}: {loc_id}\n" if args.ablate else f"***** Modular-wise Evaluation: {loc_id}\n")
    logging.info(f"{graph_dpath}\n{osm_fpath}\n{utts_fpath}\n{true_results_fpath}\n{results_dpath}\n")

    if args.module == "srer" or args.module == "all":
        srer_out_fpath_full = os.path.join(os.path.expanduser("~"), "ground", f"results_full_ablate_{args.ablate}" if args.ablate else "results_full", loc_id, srer_out_fname)
        srer_out_fpath_ablate_txt = os.path.join(os.path.expanduser("~"), "ground", "results_full_ablate_text", loc_id, srer_out_fname)
        srer_out_fpath_ablate_img = os.path.join(os.path.expanduser("~"), "ground", "results_full_ablate_image", loc_id, srer_out_fname)
        srer_out_fpath_ablate_both = os.path.join(os.path.expanduser("~"), "ground", "results_full_ablate_both", loc_id, srer_out_fname)
        if not os.path.isfile(srer_out_fpath) and os.path.isfile(srer_out_fpath_full):  # same SRER output for exp_full, exp_modular and ablate text
            copy2(srer_out_fpath_full, srer_out_fpath)
        elif not os.path.isfile(srer_out_fpath) and args.ablate and os.path.isfile(srer_out_fpath_ablate_txt):  # same SRER output for ablate text and ablate image
            copy2(srer_out_fpath_ablate_txt, srer_out_fpath)
        elif not os.path.isfile(srer_out_fpath) and args.ablate and os.path.isfile(srer_out_fpath_ablate_img):
            copy2(srer_out_fpath_ablate_img, srer_out_fpath)
        elif not os.path.isfile(srer_out_fpath) and args.ablate and os.path.isfile(srer_out_fpath_ablate_both):
            copy2(srer_out_fpath_ablate_both, srer_out_fpath)
        else:
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
