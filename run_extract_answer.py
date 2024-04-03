import json

def extract_answer(text):
    import re
    match = re.findall(r"(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)+)", text)
    print(match)
    for extracted_string in match:
        # exception handling
        if extracted_string == '2/2/1/1/2/1/4/2/3/1/1/2/0/0/0':
            extracted_string = '2/2/1/1/2/1/4/2/3/1/1/2/0/0'
            
        if extracted_string[-1] == ".":
            extracted_string = extracted_string[:-1]
        extracted_list = extracted_string.rstrip().split('/')

        if len(extracted_list) == 14:
            return [float(i) for i in extracted_list]
            # exception handling
            # return [-1 if (i == ">0") else float(i) for i in extracted_list]
    return []


with open('pathology_report_response_single.json', 'r') as f:
    data = json.load(f)
    f.close()

biopsy_result = {}
surgery_result = {}
for id in data.keys():
    _data = data[id]
    newfeature_prompt = []
    is_biopsy = None
    for text_prompt in _data["text_prompt"]:
        # print(text_prompt)
        if len(extract_answer(text_prompt)) == 0:
            print(text_prompt)
            continue
        
        newfeature = extract_answer(text_prompt)
        is_biopsy = newfeature[0]
        newfeature_prompt.append(newfeature[1:])

    if is_biopsy == 1:
        biopsy_result[id] = {"text":_data["text"],
                            "text_prompt":_data["text_prompt"],
                            "newfeature_prompt":newfeature_prompt
                            }
    elif is_biopsy == 2:
        surgery_result[id] = {"text":_data["text"],
                            "text_prompt":_data["text_prompt"],
                            "newfeature_prompt":newfeature_prompt
                            }
    elif is_biopsy == 0:
        print(id, "missing for biopsy")


# with open('pathology_report_response_biopsy_single.json', 'w') as f:
#     json.dump(biopsy_result, f, indent=4)
#     f.close()

# with open('pathology_report_response_surgery_single.json', 'w') as f:
#     json.dump(surgery_result, f, indent=4)
#     f.close()

print(len(data.keys()))
print(len(biopsy_result.keys()))
print(len(surgery_result.keys()))
print(len(data.keys())-len(biopsy_result.keys())-len(surgery_result.keys()))