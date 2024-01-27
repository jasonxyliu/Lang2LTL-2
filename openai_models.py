import os
import base64
import json
from time import sleep
import logging

import openai
from openai import OpenAI


openai.api_key = os.getenv("OPENAI_API_KEY")


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


class TextEmbedding():
    def __init__(self):
        self.client = OpenAI()

    def embed(self, txt):
        txt = json.dumps(txt).replace("\n", " ")
        complete = False
        ntries = 0
        while not complete:
            try:
                raw_responses = self.client.embeddings.create(
                    model = "text-embedding-3-large",
                    input=txt
                )
                complete = True
            except:
                sleep(30)
                logging.info(f"{ntries}: waiting for the server. sleep for 30 sec...")
                # logging.info(f"{ntries}: waiting for the server. sleep for 30 sec...\n{query_prompt}")
                logging.info("OK continue")
                ntries += 1

        embedding = raw_responses.data[0].embedding
        return embedding


def get_embed(txt):
    client = OpenAI()
    txt = json.dumps(txt).replace("\n", " ")
    complete = False
    ntries = 0
    while not complete:
        try:
            raw_responses = client.embeddings.create(
                model = "text-embedding-ada-002",
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


def prompt2msg(query_prompt):
    """
    Make prompts for GPT-3 compatible with GPT-3.5 and GPT-4.
    Support prompts for
        RER: e.g., data/osm/rer_prompt_16.txt
        symbolic translation: e.g., data/prompt_symbolic_batch12_perm/prompt_nexamples1_symbolic_batch12_perm_ltl_formula_9_42_fold0.txt
        end-to-end translation: e.g., data/osm/osm_full_e2e_prompt_boston_0.txt
    :param query_prompt: prompt used by text completion API (text-davinci-003).
    :return: message used by chat completion API (gpt-3, gpt-3.5-turbo).
    """
    # prompt_splits = query_prompt.split("\n\n")
    # system_prompt = "\n\n".join(prompt_splits[0: -1])  # task description and common examples
    # query = prompt_splits[-1]  # specific context info and query question
    #
    # msg = [{"role": "system", "content": system_prompt}]
    # msg.append({"role": "user", "content": query})

    prompt_splits = query_prompt.split("\n\n")
    task_description = prompt_splits[0]
    examples = prompt_splits[1: -1]
    query = prompt_splits[-1]

    msg = [{"role": "system", "content": task_description}]
    for example in examples:
        if "\n" in example:
            example_splits = example.split("\n")
            q = '\n'.join(example_splits[0:-1])  # every line except the last in 1 example block
            a_splits = example_splits[-1].split(" ")  # last line is the response
            q += f"\n{a_splits.pop(0)}"
            a = " ".join(a_splits)
            msg.append({"role": "user", "content": q})
            msg.append({"role": "assistant", "content": a})
        else:  # info should be in system prompt, e.g., landmark list
            msg[0]["content"] += f"\n{example}"
    msg.append({"role": "user", "content": query})

    return msg
