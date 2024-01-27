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

    breakpoint()

    response = raw_response.choices[0].message.content
    response.replace("\"", "").split(': ')[1]
    return response
