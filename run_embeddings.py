import os
from openai import OpenAI
import json

OPENAI_API_KEY = "..."

client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text, model="text-embedding-3-large"): # text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


with open('pathology_report_response_surgery_single.json', 'r') as f:
    data = json.load(f)
    f.close()

result = {}
for id in data.keys():
    _data = data[id]
    print(num_tokens_from_string(_data["text"], "cl100k_base"))

    result[id] = {"text":_data["text"], 
                  "text_prompt":_data["text_prompt"],
                  "newfeature_prompt":_data["newfeature_prompt"],
                  "embed_raw":get_embedding(_data["text"]) if num_tokens_from_string(_data["text"], "cl100k_base")<8191 else [], 
                  "embed_prompt":[get_embedding(text_prompt) for text_prompt in _data["text_prompt"]]}


with open('pathology_report_response_surgery_embeddings_3large_single.json', 'w') as f:
    json.dump(result, f, indent=4)
    f.close()