from openai_models import extract


def parse_llm_output(raw_out):
    parsed_out = {}
    for line in raw_out.split('\n'):
        if line.startswith('Referring Expressions:'):
            parsed_out["sres"] = eval(line.split('Referring Expressions: ')[1])
        elif line.startswith('Lifted Command:'):
            parsed_out["lifted_utt"] = eval(line.split('Lifted Command: ')[1])
        elif line.startswith('Symbol Map:'):
            parsed_out["lifted_symbol_map"] = eval(line.split('Symbol Map: ')[1])
        elif line.startswith('Spatial Predicates: '):
            parsed_out["spatial_preds"] = eval(line.split('Spatial Predicates: ')[1])

    # Map each referring expression to its corresponding spatial predicate:
    parsed_out["sre_to_preds"] = {}

    for sre in parsed_out["sres"]:
        found_re = False  # there may be referring expression (RE) without spatial relation

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

        if not found_re:  # find a referring expression (RE) without spatial relation
            parsed_out["sre_to_preds"][sre] = {}

    return parsed_out


def srer(command):
    raw_out = extract(command)
    parsed_out = {"utt": command}
    parsed_out.update(parse_llm_output(raw_out))
    return raw_out, parsed_out


if __name__ == "__main__":
    commands = [
        "go to the couch in front of the TV, the couch on the left of the kitchen counter, the table next to the door, the table in front of the whiteboard, and the table on the left of the bookshelf",

    ]

    for command in commands:
        raw_out, parsed_out = srer(command)

        print(f"{parsed_out['lifted_utt']}\n{parsed_out['spatial_preds']}")
        breakpoint()
