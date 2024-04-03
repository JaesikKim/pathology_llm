import json
with open("pathology_report.json", 'r') as f: # load your pathology_report file here
    data = json.load(f)

import os
from openai import OpenAI

import pandas as pd

OPENAI_API_KEY = "..."

client = OpenAI(api_key=OPENAI_API_KEY)

# def get_embedding(text, model="text-embedding-ada-002"):
#     text = text.replace("\n", " ")
#     return client.embeddings.create(input = [text], model=model).data[0].embedding

# import tiktoken

# def num_tokens_from_string(string: str, encoding_name: str) -> int:
#     encoding = tiktoken.get_encoding(encoding_name)
#     num_tokens = len(encoding.encode(string))
#     return num_tokens

def call_GPTcompletions(M_stage, text, seed=0):
    user_prompt = '''
    {report}

    This is a pathology report of a head and neck cancer surgery patient. Based on the contents of this report, extract clinical factors such as histologic grade, resection margin and so on.
    Provide a correct answer to the following multiple-choice questions and find the evidence and give a detailed reason together. At the end of each question, say like "Answer: 1" or "Answer: 2". If information was not found, choose the most possible suggestion or speculative answer. If no any evidence though, say "Answer: 0".
    1. A diagnostic biopsy or a curative surgical procedure
    1) diagnostic biopsy
    2) curative surgical procedure
    2. Histologic grade according to WHO classification
    1) G1
    2) G2
    3) G3
    3. Resection margin
    1) clear
    2) closed
    3) positive
    4. Lymphovascular invasion 
    1) no
    2) yes
    5. perineural invasion 
    1) no
    2) yes
    6. Extracapsular extension 
    1) no extracapsular extension
    2) microscopic extracapsular extension
    3) gross extracapsular extension
    7. Pathologic T stage. If several T stages are speculative based on other information, avoid saying 0 as much as possible and answer conservatively with speculative T stage.
    1) Tx
    2) T0
    3) T1
    4) T2
    5) T3
    6) T4a
    7) T4b
    8. Pathologic N stage. If several N stages are speculative based on other information, avoid saying 0 as much as possible and answer conservatively with speculative N stage.
    1) Nx
    2) N0
    3) N1
    4) N2
    5) N3
    9. What is Overall stage if pathologic M stage is {M_stage}. If several overall stages are speculative based on other information, avoid saying 0 as much as possible and answer conservatively with speculative overall stage.
    1) Stage I
    2) Stage II
    3) Stage III
    4) Stage IV
    10. Extranodal extension
    1) no
    2) yes
    11. Localization 
    1) Oral cavity
    2) Oropharynx
    3) Larynx
    4) Hypopharynx
    12. Histotype
    1) Basaloid
    2) keratinizing
    13. p16 statue
    1) negative
    2) positive
    14. Metastatic lymph node ratio
    say only the decimals (up to three decimal places) after calculating the ratio of the number of positive or metastatic lymph nodes by the total number of examined or dissected lymph nodes. Do not say '>0' or '0/0'.
    
    At the end, give a concatenated answer in this format
    Answer: write a concatenated answer here

    Make sure that the total number of questions are 14. The final answer should contain 14 answers in multiple-choice single digit numbers (float value or for 14th question) delimited by '/'.
    For example, 
    Answer: 2/2/3/1/1/1/0/2/1/0/0/0/1/0/0
    Answer: 2/2/1/1/1/1/0/4/0/1/0/0/1/1/0.1
    Answer: 1/1/2/0/1/0/1/2/3/3/1/0/1/0/0.06    
    Answer: 1/1/2/0/1/1/0/0/3/0/0/0/1/0/0.375
    '''
    # 14. According to NCCN guidelines, can adjuvant treatments such as post-operative chemotherapy and radiation therapy be considered based on reported clinical factors?
    # 0) no adjuvant therapy after surgery
    # 1) adjuvant therapy might be considered
    # 2) adjuvant therapy is strongly considered

    response = client.chat.completions.create(
      model="gpt-4-1106-preview", #gpt-4-0125-preview	
      messages=[
        {
          "role": "user",
          "content": user_prompt.format(M_stage=M_stage, report=text)
        },
      ],
      max_tokens=4096,
      top_p=0.1,
      seed=seed
    )
    return response.choices[0].message.content

meta = pd.read_csv("/project/kimlab_hnsc/data/text_files/processed_merged_hnsc_survival_clinical_dataset_20240204.csv")

# with open('extracted_pathology_report_response.json', 'r') as f:
#     _data = json.load(f)
# problems = list(_data.keys())[8:]

result = {}
for id in data.keys():
    if id in meta["sample_id"].tolist():
        # if id not in problems:
        #     continue
        M_stage = meta[meta["sample_id"] == id].pathologic_M.iloc[0]
        M_stage = "M1" if M_stage == 2 else "M0" if M_stage == 1 else "MX" if M_stage == 0 else "unknown"

        text = data[id]
        responses = []
        for i in range(1):
            response = call_GPTcompletions(M_stage, text, seed=i)
            responses.append(response)
        result[id] = {"text":text, 
                      "text_prompt":responses
                      }

with open('pathology_report_response_single.json', 'w') as f:
    json.dump(result, f, indent=4)
    f.close()