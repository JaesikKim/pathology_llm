import os

file_paths = []
for dir in os.listdir("/project/kimlab_hnsc/data/text_files/gdc_download_20240104_130443.298270/"):
    if os.path.exists("/project/kimlab_hnsc/data/text_files/gdc_download_20240104_130443.298270/"+dir+"/"):
        for filename in os.listdir("/project/kimlab_hnsc/data/text_files/gdc_download_20240104_130443.298270/"+dir+"/"):
            if filename[-4:] == ".PDF" or filename[-4:] == ".pdf":
                path = "/project/kimlab_hnsc/data/text_files/gdc_download_20240104_130443.298270/"+dir+"/"+filename
                file_paths.append(path)

from PyPDF2 import PdfReader

print(len(file_paths))
dic = {}
for i, path in enumerate(file_paths):
    id = path.split("/")[-1][:-4]
    reader = PdfReader(path)
    text = ""
    for j in range(len(reader.pages)):
        if len(reader.pages[j].extract_text()) > 0:
            text += reader.pages[j].extract_text() + "\n"
    if len(text) > 0:
        dic[id] = text
    else:
        print(path)
print(len(dic.keys()))

import json
with open('pathology_report.json', 'w') as f:
    json.dump(dic, f, indent=4)
    f.close()