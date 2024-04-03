# Survival Analysis with C-index

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test


# comparison between meta and new feature
meta = pd.read_csv("/project/kimlab_hnsc/data/text_files/processed_deduplicated_hnsc_with_ratio.csv")
import json

# with open("pathology_report_response_surgery_embeddings_ada002_single.json", 'r') as f: # ada002, 3small
#     _data = json.load(f)
#     f.close()
# with open("pathology_report_response_surgery_embeddings_3small_single.json", 'r') as f: # ada002, 3small
#     _data = json.load(f)
#     f.close()
with open("pathology_report_response_surgery_embeddings_3large_single.json", 'r') as f: # ada002, 3small
    _data = json.load(f)
    f.close()

meta_idx = []
new_feature = []
embeddings = []
for id in _data.keys():
    meta_ret = meta[meta["sample_id"] == id]
    if (len(meta_ret) >0) and (len(_data[id]["newfeature_prompt"][0]) == 13): # 1536 // 3072
        meta_idx.append(meta_ret.index[0])
        new_feature.append(_data[id]["newfeature_prompt"][0])
        embeddings.append(_data[id]["embed_prompt"][0])
meta = meta.loc[meta_idx].reset_index(drop=True)
print(meta.shape)
embeddings = pd.DataFrame(embeddings)

new_feature = pd.DataFrame(new_feature, columns=["histologic_grade", "resection_margin", "lymphovascular_invasion_present","perineural_invasion_present", "extracapsular_extension", 
                                                 "pathologic_T", "pathologic_N", "pathologic_stage",  
                                                 "extranodal_extension", "localization", "histotype", "HPV", "metastatic_lymph_node_ratio"])
new_feature['pathologic_T'] = new_feature['pathologic_T'].replace({1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6})
new_feature['pathologic_N'] = new_feature['pathologic_N'].replace({1:0, 2:1, 3:2, 4:3, 5:4})

# meta processing
meta['age'] = pd.cut(meta['age'], bins=3, labels=['0th-33th', '33th-66th', '66th-100th'], include_lowest=True)
meta['gender'] = pd.Categorical(meta['gender'].replace({2:'Male', 1:'Female'}), categories=['Female', 'Male'], ordered=True)
meta['alcohol'] = pd.Categorical(meta['alcohol'].replace({2:'Yes', 1:'No', 0:'NA'}), categories=['NA', 'No', 'Yes'], ordered=True)
meta['pathologic_M'] = pd.Categorical(meta['pathologic_M'].replace({2:'M1', 1:'M0 or MX', 0:'NA'}), categories=['NA', 'M0 or MX', 'M1'], ordered=True)
meta['postoperative_rx_tx'] = pd.Categorical(meta['postoperative_rx_tx'].replace({2:'Yes', 1:'No', 0:'NA'}), categories=['NA', 'No', 'Yes'], ordered=True)

# new_feature processing
new_feature["histologic_grade"] = pd.Categorical(new_feature["histologic_grade"].replace({3:'G3',2:'G2',1:'G1',0:'NA'}), categories=['NA', 'G1', 'G2', 'G3'], ordered=True)
new_feature["resection_margin"] = pd.Categorical(new_feature["resection_margin"].replace({3:'Positive',2:'Close',1:'Clear',0:'NA'}), categories=['NA', 'Clear', 'Close', 'Positive'], ordered=True)
new_feature["lymphovascular_invasion_present"] = pd.Categorical(new_feature["lymphovascular_invasion_present"].replace({2:'Yes',1:'No',0:'NA'}), categories=['NA', 'No', 'Yes'], ordered=True)
new_feature["perineural_invasion_present"] = pd.Categorical(new_feature["perineural_invasion_present"].replace({2:'Yes',1:'No',0:'NA'}), categories=['NA', 'No', 'Yes'], ordered=True)
new_feature["extracapsular_extension"] = pd.Categorical(new_feature["extracapsular_extension"].replace({3:'Gross extension',2:'Microscopic extension',1:'No extracapsular extension',0:'NA'}), categories=['NA', 'No extracapsular extension', 'Microscopic extension', 'Gross extension'], ordered=True)
new_feature["pathologic_T"] = pd.Categorical(new_feature["pathologic_T"].replace({6:'T4b',5:'T4a',4:'T3',3:'T2',2:'T1',1:'T0',0:'TX or NA'}), categories=['TX or NA', 'T1', 'T2', 'T3', 'T4a', 'T4b'], ordered=True)
new_feature["pathologic_N"] = pd.Categorical(new_feature["pathologic_N"].replace({4:'N3',3:'N2',2:'N1',1:'N0',0:'NX or NA'}), categories=['NX or NA', 'N0', 'N1', 'N2', 'N3'], ordered=True)
new_feature["pathologic_stage"] = pd.Categorical(new_feature["pathologic_stage"].replace({4:'Stage IV',3:'Stage III',2:'Stage II',1:'Stage I',0:'NA'}), categories=['NA', 'Stage I', 'Stage II', 'Stage III', 'Stage IV'], ordered=True)
new_feature["localization"] = pd.Categorical(new_feature["localization"].replace({4:'Hypopharynx',3:'Larynx',2:'Oropharynx',1:'Oral cavity',0:'NA'}), categories=['NA', 'Oral cavity', 'Oropharynx', 'Larynx', 'Hypopharynx'], ordered=True)
new_feature["HPV"] = pd.Categorical(new_feature["HPV"].replace({2:'Positive',1:'Negative',0:'NA'}), categories=['NA', 'Negative', 'Positive'], ordered=True)
new_feature['extranodal_extension'] = pd.Categorical(new_feature['extranodal_extension'].replace({2:'Yes', 1:'No', 0:'NA'}), categories=['NA', 'No', 'Yes'], ordered=True)
new_feature['histotype'] = pd.Categorical(new_feature['histotype'].replace({2:'Keratinizing', 1:'Basaloid', 0:'NA'}), categories=['NA', 'Basaloid', 'Keratinizing'], ordered=True)
new_feature['metastatic_lymph_node_ratio'] = pd.cut(new_feature['metastatic_lymph_node_ratio'], bins=3, labels=['0th-33th', '33th-66th', '66th-100th'], include_lowest=True)


# embeddings processing
from sklearn.decomposition import PCA
num_pc = 10
pca = PCA(n_components=num_pc)
embeddings_transformed = pca.fit_transform(embeddings)
print(pca.explained_variance_ratio_.cumsum())
embeddings_transformed = pd.DataFrame(embeddings_transformed, columns=["embedding_pc"+str(i) for i in range(num_pc)])
for column in embeddings_transformed.columns:
    embeddings_transformed[column] = pd.cut(embeddings_transformed[column], bins=3, labels=['1st tertile', '2nd tertile', '3rd tertile'], include_lowest=True)


data = pd.concat((meta[["age", "gender", "alcohol", "smoking", "pathologic_M", "postoperative_rx_tx"]], new_feature, embeddings_transformed), axis=1)

# OS
target = "OS"
_data = pd.concat((data, meta[[target, target+"_censor"]]), axis=1).dropna()

features = ['embedding_pc3', 'embedding_pc4', 'embedding_pc8']
fig, axes = plt.subplots(1, 3, figsize=(20, 6))  # Adjust the size as needed
axes = axes.flatten()
for i, feature in enumerate(features):
    kmf = KaplanMeierFitter()
    list_grouped_df = []
    for name, grouped_df in _data.groupby(feature, observed=False):
        kmf.fit(grouped_df[target], grouped_df[target+"_censor"], label=name)
        kmf.plot_survival_function(ax=axes[i])
        list_grouped_df.append((name, grouped_df))
    axes[i].set_title(feature, fontsize=20)
    axes[i].tick_params(axis='x', labelsize=20)
    axes[i].tick_params(axis='y', labelsize=20)
    axes[i].set_xlabel('Timeline', fontdict={'fontsize': 20})
    axes[i].legend(fontsize=15) 

    print(feature)
    results = multivariate_logrank_test(_data[target], _data[feature], _data[target+"_censor"])
    p = results.summary.loc[0, 'p']
    # 0.0001 0.001 0.01 0.05
    if p <= 0.0001:
        print(p, "****")
    elif p > 0.0001 and p <= 0.001:
        print(p, "***")
    elif p > 0.001 and p <= 0.01:
        print(p, "**")
    elif p > 0.01 and p <= 0.05:
        print(p, "*")
    else:
        print(p, "ns")        

plt.tight_layout()

# axes[2].axis('off')  

plt.savefig("fig/fig4_1.png")
plt.close()



# DSS
target = "DSS"
_data = pd.concat((data, meta[[target, target+"_censor"]]), axis=1).dropna()

features = ['embedding_pc4', 'embedding_pc8']

fig, axes = plt.subplots(1, 3, figsize=(20, 6))  # Adjust the size as needed
axes = axes.flatten()
for i, feature in enumerate(features):
    kmf = KaplanMeierFitter()
    list_grouped_df = []
    for name, grouped_df in _data.groupby(feature, observed=False):
        if (name == "NA") or (name == "NX or NA") or (name == "TX or NA"):
            continue
        kmf.fit(grouped_df[target], grouped_df[target+"_censor"], label=name)
        kmf.plot_survival_function(ax=axes[i])
        list_grouped_df.append((name, grouped_df))
    axes[i].set_title(feature, fontsize=20)
    axes[i].tick_params(axis='x', labelsize=20)
    axes[i].tick_params(axis='y', labelsize=20)
    axes[i].set_xlabel('Timeline', fontdict={'fontsize': 20})
    axes[i].legend(fontsize=15) 

    print(feature)
    results = multivariate_logrank_test(_data[target], _data[feature], _data[target+"_censor"])
    p = results.summary.loc[0, 'p']
    # 0.0001 0.001 0.01 0.05
    if p <= 0.0001:
        print(p, "****")
    elif p > 0.0001 and p <= 0.001:
        print(p, "***")
    elif p > 0.001 and p <= 0.01:
        print(p, "**")
    elif p > 0.01 and p <= 0.05:
        print(p, "*")
    else:
        print(p, "ns")                   

plt.tight_layout()

axes[2].axis('off')  

plt.savefig("fig/fig4_2.png")
plt.close()


# # PFS
# target = "PFS"
# _data = pd.concat((data, meta[[target, target+"_censor"]]), axis=1).dropna()

# features = ['embedding_pc4']

# fig, axes = plt.subplots(1, 3, figsize=(20, 6))  # Adjust the size as needed
# axes = axes.flatten()
# for i, feature in enumerate(features):
#     kmf = KaplanMeierFitter()
#     list_grouped_df = []
#     for name, grouped_df in _data.groupby(feature, observed=False):
#         if (name == "NA") or (name == "NX or NA") or (name == "TX or NA"):
#             continue
#         kmf.fit(grouped_df[target], grouped_df[target+"_censor"], label=name)
#         kmf.plot_survival_function(ax=axes[i])
#         list_grouped_df.append((name, grouped_df))
#     axes[i].set_title(feature, fontsize=20)
#     axes[i].tick_params(axis='x', labelsize=20)
#     axes[i].tick_params(axis='y', labelsize=20)
#     axes[i].set_xlabel('Timeline', fontdict={'fontsize': 20})
#     axes[i].legend(fontsize=15) 

#     import itertools
#     print(feature)
#     for group1, group2 in list(itertools.combinations(list_grouped_df, 2)):
#         results = logrank_test(group1[1][target], group2[1][target],
#                             event_observed_A=group1[1][target+"_censor"], event_observed_B=group2[1][target+"_censor"])
#         p = results.summary.loc[0, 'p']
#         # 0.0001 0.001 0.01 0.05
#         if p <= 0.0001:
#             print(group1[0], group2[0], p, "****")
#         elif p > 0.0001 and p <= 0.001:
#             print(group1[0], group2[0], p, "***")
#         elif p > 0.001 and p <= 0.01:
#             print(group1[0], group2[0], p, "**")
#         elif p > 0.01 and p <= 0.05:
#             print(group1[0], group2[0], p, "*")
#         else:
#             print(group1[0], group2[0], p, "ns")                    

# plt.tight_layout()

# # Remove the empty subplot (if any)
# axes[1].axis('off')  
# axes[2].axis('off')  

# plt.savefig("fig/fig4_3.png")
# plt.close()






# # OS
# target = "OS"
# _data = pd.concat((data, meta[[target, target+"_censor"]]), axis=1).dropna()

# features = ['age', 'embedding_pc3', 'embedding_pc4', 'embedding_pc8', 'extracapsular_extension', 'metastatic_lymph_node_ratio', 'pathologic_N', 'pathologic_T', 'pathologic_stage', 'perineural_invasion_present', 'postoperative_rx_tx', 'resection_margin']

# fig, axes = plt.subplots(4, 3, figsize=(20, 24))  # Adjust the size as needed
# axes = axes.flatten()
# for i, feature in enumerate(features):
#     kmf = KaplanMeierFitter()
#     list_grouped_df = []
#     for name, grouped_df in _data.groupby(feature, observed=False):
#         if (name == "NA") or (name == "NX or NA") or (name == "TX or NA"):
#             continue
#         kmf.fit(grouped_df[target], grouped_df[target+"_censor"], label=name)
#         kmf.plot_survival_function(ax=axes[i])
#         list_grouped_df.append((name, grouped_df))
#     axes[i].set_title(feature, fontsize=20)
#     axes[i].tick_params(axis='x', labelsize=20)
#     axes[i].tick_params(axis='y', labelsize=20)
#     axes[i].set_xlabel('Timeline', fontdict={'fontsize': 20})
#     axes[i].legend(fontsize=15) 

#     import itertools
#     print(feature)
#     for group1, group2 in list(itertools.combinations(list_grouped_df, 2)):
#         results = logrank_test(group1[1][target], group2[1][target],
#                             event_observed_A=group1[1][target+"_censor"], event_observed_B=group2[1][target+"_censor"])
#         p = results.summary.loc[0, 'p']
#         # 0.0001 0.001 0.01 0.05
#         if p <= 0.0001:
#             print(group1[0], group2[0], p, "****")
#         elif p > 0.0001 and p <= 0.001:
#             print(group1[0], group2[0], p, "***")
#         elif p > 0.001 and p <= 0.01:
#             print(group1[0], group2[0], p, "**")
#         elif p > 0.01 and p <= 0.05:
#             print(group1[0], group2[0], p, "*")
#         else:
#             print(group1[0], group2[0], p, "ns")                    

# plt.tight_layout()

# plt.savefig("fig/KM_OS.png")
# plt.close()



# # DSS
# target = "DSS"
# _data = pd.concat((data, meta[[target, target+"_censor"]]), axis=1).dropna()

# features = ['embedding_pc4', 'embedding_pc8', 'extracapsular_extension', 'metastatic_lymph_node_ratio', 'pathologic_N', 'pathologic_T', 'pathologic_stage', 'perineural_invasion_present', 'postoperative_rx_tx', 'resection_margin']


# fig, axes = plt.subplots(4, 3, figsize=(20, 24))  # Adjust the size as needed
# axes = axes.flatten()
# for i, feature in enumerate(features):
#     kmf = KaplanMeierFitter()
#     list_grouped_df = []
#     for name, grouped_df in _data.groupby(feature, observed=False):
#         if (name == "NA") or (name == "NX or NA") or (name == "TX or NA"):
#             continue
#         kmf.fit(grouped_df[target], grouped_df[target+"_censor"], label=name)
#         kmf.plot_survival_function(ax=axes[i])
#         list_grouped_df.append((name, grouped_df))
#     axes[i].set_title(feature, fontsize=20)
#     axes[i].tick_params(axis='x', labelsize=20)
#     axes[i].tick_params(axis='y', labelsize=20)
#     axes[i].set_xlabel('Timeline', fontdict={'fontsize': 20})
#     axes[i].legend(fontsize=15) 

#     import itertools
#     print(feature)
#     for group1, group2 in list(itertools.combinations(list_grouped_df, 2)):
#         results = logrank_test(group1[1][target], group2[1][target],
#                             event_observed_A=group1[1][target+"_censor"], event_observed_B=group2[1][target+"_censor"])
#         p = results.summary.loc[0, 'p']
#         # 0.0001 0.001 0.01 0.05
#         if p <= 0.0001:
#             print(group1[0], group2[0], p, "****")
#         elif p > 0.0001 and p <= 0.001:
#             print(group1[0], group2[0], p, "***")
#         elif p > 0.001 and p <= 0.01:
#             print(group1[0], group2[0], p, "**")
#         elif p > 0.01 and p <= 0.05:
#             print(group1[0], group2[0], p, "*")
#         else:
#             print(group1[0], group2[0], p, "ns")                    

# plt.tight_layout()

# # Remove the empty subplot (if any)
# axes[10].axis('off')  
# axes[11].axis('off')

# plt.savefig("fig/KM_DSS.png")
# plt.close()


# # PFS
# target = "PFS"
# _data = pd.concat((data, meta[[target, target+"_censor"]]), axis=1).dropna()

# features = ['embedding_pc4', 'extracapsular_extension', 'metastatic_lymph_node_ratio', 'pathologic_N', 'resection_margin']

# fig, axes = plt.subplots(2, 3, figsize=(20, 12))  # Adjust the size as needed
# axes = axes.flatten()
# for i, feature in enumerate(features):
#     kmf = KaplanMeierFitter()
#     list_grouped_df = []
#     for name, grouped_df in _data.groupby(feature, observed=False):
#         if (name == "NA") or (name == "NX or NA") or (name == "TX or NA"):
#             continue
#         kmf.fit(grouped_df[target], grouped_df[target+"_censor"], label=name)
#         kmf.plot_survival_function(ax=axes[i])
#         list_grouped_df.append((name, grouped_df))
#     axes[i].set_title(feature, fontsize=20)
#     axes[i].tick_params(axis='x', labelsize=20)
#     axes[i].tick_params(axis='y', labelsize=20)
#     axes[i].set_xlabel('Timeline', fontdict={'fontsize': 20})
#     axes[i].legend(fontsize=15) 

#     import itertools
#     print(feature)
#     for group1, group2 in list(itertools.combinations(list_grouped_df, 2)):
#         results = logrank_test(group1[1][target], group2[1][target],
#                             event_observed_A=group1[1][target+"_censor"], event_observed_B=group2[1][target+"_censor"])
#         p = results.summary.loc[0, 'p']
#         # 0.0001 0.001 0.01 0.05
#         if p <= 0.0001:
#             print(group1[0], group2[0], p, "****")
#         elif p > 0.0001 and p <= 0.001:
#             print(group1[0], group2[0], p, "***")
#         elif p > 0.001 and p <= 0.01:
#             print(group1[0], group2[0], p, "**")
#         elif p > 0.01 and p <= 0.05:
#             print(group1[0], group2[0], p, "*")
#         else:
#             print(group1[0], group2[0], p, "ns")                    

# plt.tight_layout()

# # Remove the empty subplot (if any)
# axes[5].axis('off')  # This effectively "hides" the axis at position (2, 3)

# plt.savefig("fig/KM_PFS.png")
# plt.close()
