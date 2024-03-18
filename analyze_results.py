"""
Modular-wise Evaluation: correct input to each module, check modular output.
"""
import os
import argparse
import logging
from collections import defaultdict

from evaluate import eval_srer, eval_reg, eval_spg, eval_lt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, default="reg", choices=["srer", "reg", "spg", "lt", "all"], help="domain name.")
    parser.add_argument("--ablate", type=str, default=None, choices=["both", "image", "text", None], help="ablate out a modality.")
    parser.add_argument("--nsamples", type=int, default=None, help="number of sample utts per LTL formula or None for all.")
    parser.add_argument("--topk", type=int, default=10, help="top k most likely landmarks grounded by REG.")
    parser.add_argument("--lt", type=str, default="t5", choices=["t5", "rag"], help="lifted translation model.")
    parser.add_argument("--nexamples", type=int, default=2, help="number of in-context examples if use RAG lifted translation model.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(os.path.expanduser("~"), "ground", f"analyze_results_module_{args.module}.log"), mode='w'),
                            logging.StreamHandler()
                        ]
    )
    logging.info(f"***** Analyze Modular Evaluation Results: {args.module}\n")

    metric2ncorrects, metric2total = defaultdict(int), defaultdict(int)  # SRER: nprops vs. acc; REG: RE length vs. top-10 acc

    for loc in ["providence", "auckland", "boston", "san_francisco"]:
        for seed in [0, 1, 2, 42, 111 ]:
            loc_id = f"{loc}_n{args.nsamples}_seed{seed}" if args.nsamples else f"{loc}_all_seed{seed}"
            lt_id = f"lt-{args.lt}{args.nexamples}" if args.lt == "rag" else f"{args.lt}"

            data_dpath = os.path.join(os.path.expanduser("~"), "ground", "data")
            results_dname = "results_modular_ablate_both" if seed == 111 else "results_modular"
            results_dpath = os.path.join(os.path.expanduser("~"), "ground", results_dname, loc_id)
            srer_out_fname = "srer_outs.json"
            srer_out_fpath = os.path.join(results_dpath, srer_out_fname)
            reg_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "reg"))
            spg_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", "spg"))
            lt_out_fpath = os.path.join(results_dpath, srer_out_fname.replace("srer", f"lt-{lt_id}"))
            true_results_dname = f"{loc}_ablate" if seed == 111 else f"{loc}"
            true_results_fpath = os.path.join(data_dpath, "dataset", true_results_dname, f"{loc_id}_true_results.json")

            if args.module == "srer" or args.module == "all":
                nprops2acc = eval_srer(true_results_fpath, srer_out_fpath)
                for nprops, (ncorrects, ntotal) in nprops2acc.items():
                    metric2ncorrects[nprops] += ncorrects
                    metric2total[nprops] += ntotal

            if args.module == "reg" or args.module == "all":
                len2acc = eval_reg(true_results_fpath, args.topk, reg_out_fpath)
                for re_len, (ncorrects, ntotal) in len2acc.items():
                    metric2ncorrects[re_len] += ncorrects
                    metric2total[re_len] += ntotal

            if args.module == "spg" or args.module == "all":
                len2acc = eval_spg(true_results_fpath, args.topk, spg_out_fpath)

            if args.module == "lt" or args.module == "all":
                eval_lt(true_results_fpath, lt_out_fpath)

    if args.module == "srer":
        nprops2acc = {nprops: ncorrects / metric2total[nprops] for nprops, ncorrects in metric2ncorrects.items()}
        logging.info(f"SRER nprops vs. acc: {nprops2acc}")

    if args.module == "reg":
        len2acc = {re_len: ncorrects / metric2total[re_len] for re_len, ncorrects in metric2ncorrects.items()}
        len2acc_sorted = sorted(len2acc.items(), key=lambda kv: kv[0])
        len2acc_sorted = {len: acc for len, acc in len2acc_sorted}
        logging.info(f"REG RE length vs. acc: {len2acc_sorted}")
