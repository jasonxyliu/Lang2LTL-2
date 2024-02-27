import os
import base64
import json
from time import sleep
import logging

import openai
from openai import OpenAI


openai.api_key = os.getenv("OPENAI_API_KEY")


def extract(command):
    client = OpenAI()
    raw_responses = client.chat.completions.create(
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
                    # NOTE: below are the in-context examples:
                    "\n\nHere are some examples:" \
                    "\n\nCommand: \"move to the red room\"\nEntities: [\"the red room\"]\nReferring Expressions: [\"the red room\"]\nSpatial Predicates: []\nReferring Expression to Predicates Map: {\"the red room\": {}}\nLifted Command: \"move to a\"\nSymbol Map: {\"a\": \"the red room}" \
                    "\n\nCommand: \"Walk to the north of Subway\"\nEntities: [\"Subway\"]\nReferring Expressions: [\"the north of Subway\"]\nSpatial Predicates [{\"north of\": [\"Subway\"]}]\nReferring Expression to Predicates Map: {\"the north of Subway\": {\"north of\": [\"Subway\"]}}\nLifted Command: \"Walk to a\"\nSymbol Map: {\"a\": \"the north of Subway\"}" \
                    "\n\nCommand: \"go to the bicycle rack in front of the CIT\"\nEntities: [\"the bicycle rack\", \"the CIT\"]\nReferring Expressions: [\"the bicycle rack in front of the CIT\"]\nSpatial Predicates: [{\"in front of\": [\"the bicycle rack\", \"the CIT\"]}]\nReferring Expression to Predicates Map: {\"the bicycle rack in front of the CIT\": {\"in front of\": [\"the bicycle rack\", \"the CIT\"]}}\nLifted Command: \"go to a\"\nSymbol Map: {\"a\": \"the bicycle rack in front of the CIT\"}" \
                    "\n\nCommand: \"go through the red room to the blue room or the yellow room to the green room but do not go in the purple room\"\nEntities: [\"the red room\", \"the blue room\", \"the yellow room\", \"the green room\", \"the purple room\"]\nReferring Expressions: [\"the red room\", \"the blue room\", \"the yellow room\", \"the green room\", \"the purple room\"]\nSpatial Predicates: []\nReferring Expression to Predicates Map: {\"the red room\": {}, \"the blue room\": {}, \"the yellow room\": {}, \"the green room\": {}, \"the purple room\": {}}\nLifted Command: \"go through a to b or c to d but do not go in h\"\nSymbol Map: {\"a\": \"the red room\", \"b\": \"the blue room\", \"c\": \"the yellow room\", \"d\": \"the green room\", \"h\": \"the purple room\"}" \
                    "\n\nCommand: \"Visit The Kensington, HI Boston, and Dunkin' Donuts in that specific order. make sure not to visit waypoints out of turn\"\nEntities: [\"The Kensington\", \"HI Boston\", \"Dunkin' Donuts\"]\nReferring Expressions: [\"The Kensington\", \"HI Boston\", \"Dunkin' Donuts\"]\nSpatial Predicates: []\nReferring Expression to Predicates Map: {\"The Kensington\", \"HI Boston\", \"Dunkin' Donuts\"]\nReferring Expressions: [\"The Kensington\": {}, \"HI Boston\": {}, \"Dunkin' Donuts\": {}}\nLifted Command: \"Visit a, b, and c in that specific order. make sure not to visit waypoints out of turn\"\nSymbol Map: {\"a\": \"The Kensington\",  \"b\": \"HI Boston\", \"c\": \"Dunkin' Donuts\"}" \
                    "\n\nCommand: \"do not go to the bench in front of SciLi until you visit the bicycle rack in front of the CIT\"\nEntities: [\"the bench\",  \"SciLi\", \"the bicycle rack\", \"the CIT\"]\nReferring Expressions: [\"the bench in front of SciLi\", \"the bicycle rack in front of the CIT\"]\nSpatial Predicates: [{\"in front of\": [\"the bench\", \"SciLi\"]}, {\"in front of\": [\"the bicycle rack, \"the CIT\"]}]\nReferring Expression to Predicate Map: {\"the bench in front of SciLi\": {\"in front of\": [\"the bench\", \"SciLi\"]} \"the bicycle rack in front of the CIT\": {\"in front of\": [\"the bicycle rack, \"the CIT\"]}}\nLifted Command: \"do not go to a until you visit b\"\nSymbol Map: {\"a\": \"the bench in front of SciLi\", \"b\": \"the bicycle rack in front of the CIT\"}"
            },
            {
                "role": "user",
                "content": f"Extract the referring expressions to predicates map, lifted command, and symbol map for the following command:\n\nCommand:{command}"
            }
        ],
    )
    return raw_responses.choices[0].message.content


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class GPT4V:
    def __init__(self, temp=0, max_tokens=128, n=1, stop=['\n']):
        self.client = OpenAI()

        self.temp = temp
        self.max_tokens = max_tokens
        self.n = n
        self.stop = stop

    def caption(self, img_fpath):
        complete = False
        ntries = 0
        while not complete:
            try:
                raw_responses = self.client.chat.completions.create(
                    model = "gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "What's the most obivous object in this image in one sentence."},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_fpath)}"},
                                },
                            ],
                        }
                    ],
                    max_tokens=3000,
                    # temperature=self.temp,
                    # n=self.n,
                    # stop=self.stop,
                    # max_tokens=self.max_tokens,
                )
                complete = True
            except:
                logging.info(f"{ntries}: waiting for the server. sleep for 30 sec...")
                sleep(30)
                logging.info("OK continue")
                ntries += 1
        # if self.n == 1:
        #     responses = [raw_responses["choices"][0]["message"]["content"].strip()]
        # else:
        #     responses = [choice["message"]["content"].strip() for choice in raw_responses["choices"]]

        return raw_responses.choices[0].message.content


def get_embed(txt):
    client = OpenAI()
    txt = json.dumps(txt).replace("\n", " ")
    complete = False
    ntries = 0
    while not complete:
        try:
            raw_responses = client.embeddings.create(
                model = "text-embedding-3-large",
                input=txt
            )
            complete = True
        except:
            sleep(30)
            print(f"{ntries}: waiting for the server. sleep for 30 sec...")
            # logging.info(f"{ntries}: waiting for the server. sleep for 30 sec...\n{query_prompt}")
            print("OK continue")
            ntries += 1

    embedding = raw_responses.data[0].embedding
    return embedding


def translate(query, examples):
    client = OpenAI()

    complete = False
    ntries = 0
    task = "You are an expert at translating natural language commands to linear temporal logic (LTL) formulas."
    while not complete:
        try:
            raw_response = client.chat.completions.create(
                model = "gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": f"{task}\n\nHere are some examples:\n\n{examples}"
                    },
                    {
                        "role": "user",
                        "content": f"Translate the following command to an LTL formula\n\nCommand: \"{query}\""
                    }
                ],
                temperature=0.1,
                max_tokens=1000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            complete = True
        except:
            sleep(30)
            print(f"{ntries}: waiting for the server. sleep for 30 sec...")
            # logging.info(f"{ntries}: waiting for the server. sleep for 30 sec...\n{query_prompt}")
            print("OK continue")
            ntries += 1

    response = raw_response.choices[0].message.content
    print(f"GPT query: {query}\n{response}")
    response = response.replace("\"", "").split(': ')[1]
    print(response)

    breakpoint()
    return response
