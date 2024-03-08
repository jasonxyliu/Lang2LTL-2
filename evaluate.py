import logging
from collections import defaultdict
import string
import spot

from utils import load_from_file


def eval_srer(true_results_fpath, srer_out_fpath):
    logging.info("***** Evaluating SRER Module")

    true_outs = load_from_file(true_results_fpath)
    srer_outs = load_from_file(srer_out_fpath)
    ncorrects = 0

    assert len(srer_outs) == len(true_outs), f"ERROR different numbers of samples:\ntrue: {len(true_outs)}\npred: {len(srer_outs)}"

    for true_out, srer_out in zip(true_outs, srer_outs):
        assert srer_out["utt"].strip() == true_out["utt"].strip(), f"ERROR different utterances:\ntrue: {true_out['utt']}\npred: {srer_out['utt']}"
        logging.info(f"* Command: {srer_out['utt']}")
        is_correct = True

        if len(srer_out["sre_to_preds"]) != len(true_out["sre_to_preds"]):
            is_correct = False
            logging.info(f"ERROR incorrect number of spatial predicates\ntrue: {true_out['sre_to_preds']}\npred: {srer_out['sre_to_preds']}")

        for sre_out, preds_out in srer_out["sre_to_preds"].items():
            if sre_out not in true_out["sre_to_preds"]:
                is_correct = False
                logging.info(f"ERROR incorrect SRE:\ntrue: {list(true_out['sre_to_preds'].keys())}\nnot contain pred: {sre_out}")
            else:
                preds_true = true_out["sre_to_preds"][sre_out]

                if list(preds_true.keys())[0] == "None" and preds_out:  # referring expression with spatial relation
                    logging.info(f"ERROR incorrect spatial predicate:\ntrue: {preds_true}\nnot contain pred: {preds_out}")

                for (rel_true, res_true), (rel_out, res_out) in zip(preds_true.items(), preds_out.items()):
                    if rel_out.strip() != rel_true.strip() and rel_out not in rel_true:  # e.g., pred: left of; true: to the left of
                        is_correct = False
                        logging.info(f"ERROR incorrect spatial relation\ntrue: {rel_true}\npred: {rel_out}")

                    res_out_lower = [re_true.lower() for re_true in res_out]  # output lowercase e.g., italian resturant
                    res_true_lower = [re_true.lower() for re_true in res_true]
                    if not (len(res_out) == len(res_true) and set(res_out_lower) == set(res_true_lower)):
                        is_correct = False
                        logging.info(f"ERROR incorrect REs\ntrue: {res_true}\npred: {res_out}\n true lower: {res_true_lower}\npred lower: {res_out_lower}")

        true_lifted_utt = true_out["lifted_utt"].strip().translate(str.maketrans('', '', string.punctuation))
        srer_lifted_utt = srer_out["lifted_utt"].strip().translate(str.maketrans('', '', string.punctuation))
        if srer_lifted_utt != true_lifted_utt:
            logging.info(f"WARNING lifted commands do not exactly match\ntrue: {true_out['lifted_utt']}\npred: {srer_out['lifted_utt']}")
            if len(true_lifted_utt) != len(srer_lifted_utt):
                is_correct = False
                logging.info(f"ERROR incorrect lifted utterances\ntrue: {true_out['lifted_utt']}\npred: {srer_out['lifted_utt']}")
            else:
                # NOTE: whitespace check to make sure the lifted utterances are equivalent:
                whitespaces_srer = [i for i, letter in enumerate(srer_lifted_utt) if letter == ' ']
                whitespaces_true = [i for i, letter in enumerate(true_lifted_utt) if letter == ' ']
                if whitespaces_srer != whitespaces_true:
                    is_correct = False
                    logging.info(f"ERROR Non-matching whitespaces:\ntrue: {true_out['lifted_utt']}\npred: {srer_out['lifted_utt']}")

        if is_correct:
            ncorrects += 1
        else:
            logging.info("Incorrect SRER output")

        logging.info("\n")
    logging.info(f"SRER Accuracy: {ncorrects}/{len(true_outs)} = {ncorrects / len(true_outs)}\n\n")


def eval_reg(true_results_fpath, topk, reg_out_fpath):
    """
    Compute the top K accuracy of Referring Expression Grounding module.
    """
    logging.info("***** Evaluating REG Module")
    true_outs = load_from_file(true_results_fpath)
    reg_outs = load_from_file(reg_out_fpath)
    topk2acc = defaultdict(int)
    total_res = 0

    assert len(reg_outs) == len(true_outs), f"ERROR different numbers of samples\ntrue: {len(true_outs)}\npred: {len(reg_outs)}"

    for true_out, reg_out in zip(true_outs, reg_outs):
        assert reg_out["utt"] == true_out["utt"], f"ERROR different utterances:\ntrue: {true_out['utt']}\npred: {reg_out['utt']}"
        logging.info(f"* Command: {true_out['utt']}")

        true_ground_sre_to_preds = true_out["grounded_sre_to_preds"]
        reg_ground_sre_to_preds = reg_out["grounded_sre_to_preds"]
        if len(reg_ground_sre_to_preds) != len(true_ground_sre_to_preds):
            logging.info(f"ERROR incorrect number of spatial referring expression:\ntrue: {true_ground_sre_to_preds}\npred: {reg_ground_sre_to_preds}")

        for sre_out, pred_out in reg_ground_sre_to_preds.items():
            total_res += len(list(pred_out.values())[0])

            if sre_out not in true_ground_sre_to_preds:
                logging.info(f"ERROR incorrect SRE:\ntrue: {list(true_ground_sre_to_preds.keys())}\nnot contain pred: {sre_out}")
                continue
            else:
                pred_true = true_ground_sre_to_preds[sre_out]

                if len(pred_out) != len(pred_true):
                    logging.info(f"ERROR incorrect size of spatial predicate:\ntrue: {len(pred_true)}\npred: {len(pred_out)}")
                    continue

                res_true = [score_re[0][1] for score_re in list(pred_true.values())[0]]
                res_out = [[score_ground[1] for score_ground in grounded_res] for grounded_res in list(pred_out.values())[0]]

                for re_true, res_topk in zip(res_true, res_out):
                    for end_idx in range(1, topk+1):
                        if re_true in res_topk[:end_idx]:
                            topk2acc[end_idx] += 1
                        else:
                            if end_idx == topk:
                                logging.info(f"Incorrect Top-{topk} REG: \n{sre_out}\ntrue: {re_true}\npred: {res_topk}")
        logging.info("\n")

    for idx in range(1, topk+1):
        logging.info(f"REG Top-{idx} Accuracy: {topk2acc[idx]} / {total_res} = {topk2acc[idx] / total_res}")
    logging.info("\n\n")


def eval_spg(true_results_fpath, topk, spg_out_fpath):
    """
    Compute the top K accuracy of Spatial Predicate Grounding module.
    """
    logging.info("***** Evaluating SPG Module")

    true_outs = load_from_file(true_results_fpath)
    spg_outs = load_from_file(spg_out_fpath)
    topk2acc = defaultdict(int)
    total_sps = 0

    assert len(spg_outs) == len(true_outs), f"ERROR different numbers of samples\ntrue: {len(true_outs)}\npred: {len(spg_outs)}"

    for true_out, spg_out in zip(true_outs, spg_outs):
        assert spg_out["utt"] == true_out["utt"], f"ERROR different utterances:\ntrue: {true_out['utt']}\npred: {spg_out['utt']}"
        logging.info(f"* Command: {true_out['utt']}")

        total_sps += len(true_out["grounded_sps"])

        true_ground_sps = true_out["grounded_sps"]
        spg_ground_sps = spg_out["grounded_sps"]
        if len(spg_ground_sps) != len(true_ground_sps):
            logging.info(f"ERROR incorrect number of spatial referring expression:\ntrue: {true_ground_sps}\npred: {spg_ground_sps}")

        for sre_out, sps_topk_out in spg_ground_sps.items():
            if sre_out not in true_ground_sps:
                logging.info(f"ERROR incorrect SRE:\ntrue: {list(true_ground_sps.keys())}\nnot contain pred: {sre_out}")
                continue
            else:
                sp_true = true_ground_sps[sre_out][0]

                for end_idx in range(1, topk+1):
                    for sp_out in sps_topk_out[:end_idx]:
                        if len(sp_true) != len(sp_out):
                            logging.info(f"ERROR spatial predicates have different sizes\n{sre_out}\ntrue: {sp_true}\npred: {sp_out}")
                            continue

                        is_correct = True
                        for (lmk_type_true, ground_true), (lmk_type_out, ground_out) in zip(sp_true.items(), sp_out.items()):
                            if lmk_type_out != lmk_type_true or ground_out != ground_true:
                            # if lmk_type_out != lmk_type_true or not (set(ground_out) & set(ground_true)):
                                is_correct = False
                                if end_idx == 1:
                                    logging.info(f"Incorrect Top-1 SPG: \n{sre_out}\ntrue: ({lmk_type_true}) {ground_true}\npred: ({lmk_type_out}) {ground_out}")

                        if is_correct:
                            topk2acc[end_idx] += 1
        logging.info("\n")

    for idx in range(1, topk+1):
        logging.info(f"SPG Top-{idx} Accuracy: {topk2acc[idx]} / {total_sps} = {topk2acc[idx] / total_sps}")
    logging.info("\n\n")


def eval_lt(true_results_fpath, lt_out_fpath):
    logging.info("***** Evaluating LT")

    true_outs = load_from_file(true_results_fpath)
    lt_outs = load_from_file(lt_out_fpath)
    ncorrects = 0

    assert len(lt_outs) == len(true_outs), f"ERROR different numbers of samples\ntrue: {len(true_outs)}\npred: {len(lt_outs)}"

    for true_out, lt_out in zip(true_outs, lt_outs):
        assert lt_out["utt"] == true_out["utt"], f"ERROR different utterances:\ntrue: {true_out['utt']}\npred: {lt_out['utt']}"
        logging.info(f"* Command: {lt_out['utt']}")

        is_correct = True
        ltl_true, ltl_out = spot.formula(true_out["lifted_ltl"]), spot.formula(lt_out["lifted_ltl"])

        try:
            spot_correct = spot.are_equivalent(ltl_true, ltl_out)

            if not spot_correct:  # invariant to order of propositions
                ltl_str_true, ltl_str_out = str(ltl_true), str(ltl_out)
                prop2sre_true, prop2sre_out = {}, {}

                for prop, sre in zip(true_out["props"], true_out["sre_to_preds"].keys()):
                    ltl_str_true = ltl_str_true.replace(prop, f"<{prop}>")
                    prop2sre_true[f"<{prop}>"] = sre
                for prop, sre in prop2sre_true.items():
                    ltl_str_true = ltl_str_true.replace(prop, sre.lower())

                if "lifted_symbol_map" in lt_out:  # exp_full input previous module
                    lifted_symbol_map = lt_out["lifted_symbol_map"]
                else:  # exp_modular input ground truth (does not have "lifted_symbol_map" key)
                    lifted_symbol_map = {prop: sre for prop, sre in zip(lt_out["props"], lt_out["sre_to_preds"].keys())}
                for prop, sre in lifted_symbol_map.items():
                    ltl_str_out = ltl_str_out.replace(prop, f"<{prop}>")
                    prop2sre_out[f"<{prop}>"] = sre
                for prop, sre in prop2sre_out.items():
                    ltl_str_out = ltl_str_out.replace(prop, sre.lower())

                spot_correct = ltl_str_out == ltl_str_true
        except SyntaxError:
            is_correct = False
            logging.info(f"ERROR incorrect lifted translation Syntax Error\ntrue: {ltl_true}\npred: {ltl_out}")

        if not spot_correct:
            is_correct = False
            logging.info(f"ERROR incorrect lifted translation:\ntrue: {spot.formula(ltl_true)}\npred: {spot.formula(ltl_out)}")

        if is_correct:
            ncorrects += 1

    logging.info(f"LT Accuracy: {ncorrects} / {len(true_outs)} = {ncorrects / len(true_outs)}\n\n")
