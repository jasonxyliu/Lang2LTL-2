import os

import openai
from openai import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")


def prompt_referring_exp(command):
    # -- use OpenAI's API for interacting with GPT (using the Chat variation):
    client = OpenAI()
    prompt = client.chat.completions.create(
        model="gpt-4",
        temperature=0.1,
        max_tokens=1500,
        frequency_penalty=0,
        presence_penalty=0,
        top_p=1,
        messages=[
            {
                "role": "system",
                "content":
                    "Your task is to extract referring expressions from a natural language command."
                    " These referring expressions correspond to entities (landmarks, objects, or other nouns), and spatial relations describing relative locations."
                    " Avoid adding temporal relations indicating time sequencing of visits to spatial relations."
                    " Create a dictionary that maps each spatial relation to its corresponding entities."
                    " Repeat the command before writing output."
                    " When generating the symbol map, do not use the letters \"e\", \"f\", or \"g\"." \
                    # NOTE: below are the incontext examples:
                    "\n\nHere are some examples:" \
                    "\n\nCommand: \"move to red room\"\nEntities: [\"red room\"]\nReferring Expressions: [\"red room\"]\nSpatial Predicates: []\nReferring Expression to Predicates Map: {\"red room\": {}}\nLifted Command: \"move to a\"\nSymbol Map: {\"a\": \"red room}" \
                    "\n\nCommand: \"Walk to the north of Subway\"\nEntities: [\"Subway\"]\nReferring Expressions: [\"the north of Subway\"]\nSpatial Predicates [{\"north of\": [\"Subway\"]}]\nReferring Expression to Predicates Map: {\"the north of Subway\": {\"north of\": [\"Subway\"]}}\nLifted Command: \"Walk to a\"\nSymbol Map: {\"a\": \"the north of Subway\"}" \
                    "\n\nCommand: \"go to the bicycle rack in front of the CIT\"\nEntities: [\"the bicycle rack\", \"the CIT\"]\nReferring Expressions: [\"the bicycle rack in front of the CIT\"]\nSpatial Predicates: [{\"in front of\": [\"the bicycle rack\", \"the CIT\"]}]\nReferring Expression to Predicates Map: {\"the bicycle rack in front of the CIT\": {\"in front of\": [\"the bicycle rack\", \"the CIT\"]}}\nLifted Command: \"go to a\"\nSymbol Map: {\"a\": \"the bicycle rack in front of the CIT\"}" \
                    "\n\nCommand: \"go through red room to blue room or yellow room to green room but do not go in purple room\"\nEntities: [\"red room\", \"blue room\", \"yellow room\", \"green room\", \"purple room\"]\nReferring Expressions: [\"red room\", \"blue room\", \"yellow room\", \"green room\", \"purple room\"]\nSpatial Predicates: []\nReferring Expression to Predicates Map: {\"red room\": {}, \"blue room\": {}, \"yellow room\": {}, \"green room\": {}, \"purple room\": {}}\nLifted Command: \"go through a to b or c to d but do not go in h\"\nSymbol Map: {\"a\": \"red room\", \"b\": \"blue room\", \"c\": \"yellow room\", \"d\": \"green room\", \"h\": \"purple room\"}" \
                    "\n\nCommand: \"Visit The Kensington, HI Boston, and Dunkin' Donuts in that specific order. make sure not to visit waypoints out of turn\"\nEntities: [\"The Kensington\", \"HI Boston\", \"Dunkin' Donuts\"]\nReferring Expressions: [\"The Kensington\", \"HI Boston\", \"Dunkin' Donuts\"]\nSpatial Predicates: []\nReferring Expression to Predicates Map: {\"The Kensington\", \"HI Boston\", \"Dunkin' Donuts\"]\nReferring Expressions: [\"The Kensington\": {}, \"HI Boston\": {}, \"Dunkin' Donuts\": {}}\nLifted Command: \"Visit a, b, and c in that specific order. make sure not to visit waypoints out of turn\"\nSymbol Map: {\"a\": \"The Kensington\",  \"b\": \"HI Boston\", \"c\": \"Dunkin' Donuts\"}" \
                    "\n\nCommand: \"do not go to the bench in front of SciLi until you visit the bicycle rack in front of the CIT\"\nEntities: [\"the bench\",  \"SciLi\", \"the bicycle rack\", \"the CIT\"]\nReferring Expressions: [\"the bench in front of SciLi\", \"the bicycle rack in front of the CIT\"]\nSpatial Predicates: [{\"in front of\": [\"the bench\", \"SciLi\"]}, {\"in front of\": [\"the bicycle rack, \"the CIT\"]}]\nReferring Expression to Predicate Map: {\"the bench in front of SciLi\": {\"in front of\": [\"the bench\", \"SciLi\"]} \"the bicycle rack in front of the CIT\": {\"in front of\": [\"the bicycle rack, \"the CIT\"]}}\nLifted Command: \"do not go to a until you visit b\"\nSymbol Map: {\"a\": \"the bench in front of SciLi\", \"b\": \"the bicycle rack in front of the CIT\"}"
            },
            {
                "role": "user",
                "content": f"Extract the referring expressions to predicates map, lifted command, and symbol map for the following command:\n\nCommand:{command}"
            }
        ],
    )

    return prompt.choices[0].message.content


def parse_LLM_output(output):
    # -- we will save all the split results into a dictionary:
    llm_output = {}
    for line in output.split('\n'):
        # -- we will check each split line from the LLM's output:
        if line.startswith('Referring Expressions:'):
            # -- adding the spatial referring expressions to the dictionary:
            output = eval(line.split('Referring Expressions: ')[1])
            llm_output['sres'] = output
        elif line.startswith('Lifted Command:'):
            # -- adding the generated lifted command to the dictionary:
            output = eval(line.split('Lifted Command: ')[1])
            llm_output['lifted_utt'] = output
        elif line.startswith('Symbol Map:'):
            # -- adding the generated symbol map to the dictionary:
            output = eval(line.split('Symbol Map: ')[1])
            llm_output['sre_map'] = output
        elif line.startswith('Spatial Predicates: '):
            output = eval(line.split('Spatial Predicates: ')[1])
            llm_output['spatial_preds'] = output

    # -- mapping each referring expression to its corresponding spatial predicates map:
    llm_output['re_to_preds'] = {}
    spatial_preds = list(llm_output['spatial_preds'])
    for re in llm_output['sres']:
        try:
            spatial_pred = spatial_preds.pop(0)
        except Exception:
            print(f"ERROR: {llm_output['spatial_preds']}")
            continue

        # -- test if the spatial predicate belongs to this referring expression:
        spatial_relation = list(spatial_pred.keys())[-1]
        spatial_args = spatial_pred[spatial_relation]

        not_match = False
        if spatial_relation not in re:
            not_match = True
        for arg in spatial_args:
            if arg not in re:
                not_match = True
        if not_match == False:
            llm_output['re_to_preds'][re] = spatial_pred

    return llm_output


def srer(command):
    raw_output = prompt_referring_exp(command)
    parsed_output = parse_LLM_output(raw_output)
    parsed_output['utt'] = command
    return raw_output, parsed_output


if __name__ == "__main__":
    commands = [
        "go to the couch in front of the TV, the couch on the left of the kitchen counter, the table next to the door, the table in front of the whiteboard, and the table on the left of the bookshelf",

    ]

    for command in commands:
        raw_out, parsed_out = srer(command)

        print(f"{parsed_out['lifted_utt']}\n{parsed_out['spatial_preds']}")

        breakpoint()