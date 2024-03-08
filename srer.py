import os
from tqdm import tqdm
import logging

from openai_models import extract
from utils import load_from_file, save_to_file


def parse_llm_output(utt, raw_out):
    parsed_out = {}
    for line in raw_out.split('\n'):
        if line.startswith("Referring Expressions:"):
            parsed_out["sres"] = eval(line.split("Referring Expressions: ")[1])
        if line.startswith("Spatial Predicates: "):
            parsed_out["spatial_preds"] = eval(line.split("Spatial Predicates: ")[1])
        if line.startswith("Lifted Command:"):
            parsed_out["lifted_utt"] = eval(line.split("Lifted Command: ")[1])

    # Map each spatial referring expression (SRE) to its corresponding spatial predicate
    parsed_out["sre_to_preds"] = {}

    for sre in parsed_out["sres"]:
        found_re = False  # there may be RE without spatial relation

        for pred in parsed_out["spatial_preds"]:
            relation, lmks = list(pred.items())[0]

            if relation in sre:
                num_matches = 0
                for lmk in lmks:
                    if lmk in sre:
                        num_matches += 1

                if len(lmks) == num_matches:
                    parsed_out["sre_to_preds"][sre] = pred
                    found_re = True

        if not found_re:  # find RE without spatial relation
            parsed_out["sre_to_preds"][sre] = {}

    # Replace spatial referring expressions by symbols
    lifted_utt = utt.lower()
    lifted_symbol_map = {}  # symbol to SRE

    # Sort SREs in reverse order of number of their spatial preds
    sre_to_preds = parsed_out["sre_to_preds"].items()
    syms = ['a', 'b', 'c', 'd', 'h', 'i', 'j'][0: len(sre_to_preds)]
    lifted_symbol_map = {sym: sre[0].lower() for sre, sym in sorted(zip(list(sre_to_preds), syms), key=lambda kv: len(kv[0][1]), reverse=True)}

    for sym, sre in (lifted_symbol_map.items()):
        lifted_utt = lifted_utt.replace(sre, sym)

    # if parsed_out["lifted_utt"] != lifted_utt:
    #     logging.info(f"{utt}\n{lifted_symbol_map}")
    #     logging.info(f"SRER lifted utt:\nLLM: {parsed_out['lifted_utt']}\nMAN: {lifted_utt}\n")
    #     breakpoint()
    parsed_out["lifted_utt"] = lifted_utt
    parsed_out["lifted_symbol_map"] = lifted_symbol_map
    return parsed_out


def srer(utt):
    raw_out = extract(utt)
    parsed_out = {"utt": utt}
    parsed_out.update(parse_llm_output(utt, raw_out))
    return raw_out, parsed_out


def run_exp_srer(utts_fpath, srer_out_fpath):
	if not os.path.isfile(srer_out_fpath):
		srer_outs = []
		utts = load_from_file(utts_fpath)
		for utt in tqdm(utts, desc="Running spatial referring expression recognition (SRER) module"):
			_, srer_out = srer(utt)
			srer_outs.append(srer_out)
		save_to_file(srer_outs, srer_out_fpath)


if __name__ == "__main__":
    utts = [
        "you cannot go to other place from the park bench by the comics store unless you see the vendor cart between the Best of Boston clothing store and Boston Pewter Company",
        "move the ATM to the right of TD Bank then find Starbucks behind the restaurant called Ned Devines",
        "visit the street vendor cart in front of the coffee shop called Cafe Pulse exactly once avoid the car at the shop called Boston Pewter Company or the bicycle rack before the street vendor cart in front of the coffee shop called Cafe Pulse then reach the car at the shop called Boston Pewter Company exactly once avoid the bicycle rack before the car at the shop called Boston Pewter Company finally move to the bicycle rack",
    ]

    for utt in utts:
        raw_out, parsed_out = srer(utt)

        print(f"{parsed_out['lifted_utt']}\n\n{parsed_out['lifted_symbol_map']}\n\n{parsed_out['sre_to_preds']}\n\n\n")
        breakpoint()
