import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# comparison between meta and new feature
meta = pd.read_csv("/project/kimlab_hnsc/data/text_files/processed_deduplicated_hnsc_with_ratio.csv")
import json
with open("pathology_report_response_surgery_embeddings_3large_single.json", 'r') as f:
    _data = json.load(f)
    f.close()
# with open("pathology_report_response_surgery_single.json", 'r') as f:
#     _data = json.load(f)
#     f.close()

print(meta.shape)

meta_idx = []
embeddings = []
new_feature = []
for id in _data.keys():
    meta_ret = meta[meta["sample_id"] == id]
    if (len(meta_ret) >0) and (len(_data[id]["newfeature_prompt"][0]) == 13): # and (len(_data[id]["embed_prompt"][0]) == 1536): # 1536 // 3072
        meta_idx.append(meta_ret.index[0])
        embeddings.append(_data[id]["embed_prompt"][0])
        new_feature.append(_data[id]["newfeature_prompt"][0])
meta = meta.loc[meta_idx].reset_index(drop=True)
print(meta.shape)
embeddings = np.array(embeddings)

new_feature = pd.DataFrame(new_feature, columns=["histologic_grade", "resection_margin", "lymphovascular_invasion_present","perineural_invasion_present", "extracapsular_extension", 
                                                 "pathologic_T", "pathologic_N", "pathologic_stage",  
                                                 "extranodal_extension", "localization", "histotype", "HPV", "metastatic_lymph_node_ratio"])
# new_feature['pathologic_TX'] = (new_feature['pathologic_T'] == 1).astype(int)
# new_feature['pathologic_NX'] = (new_feature['pathologic_N'] == 1).astype(int)
new_feature['pathologic_T'] = new_feature['pathologic_T'].replace({1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6})
new_feature['pathologic_N'] = new_feature['pathologic_N'].replace({1:0, 2:1, 3:2, 4:3, 5:4})

# meta["survival"] = "Unclear"
meta.loc[(meta["OS"] < 6) & (meta["OS_censor"] == 1), "survival"] = "Poor"
meta.loc[(meta["OS"] > 36) & (meta["OS_censor"] == 0), "survival"] = "Good"
meta.loc[(meta["OS"] > 60) & (meta["OS_censor"] == 0), "survival"] = "Very good"
meta["survival"] = pd.Categorical(meta["survival"], categories=['Poor', 'Good', 'Very good'], ordered=True)

print(meta['survival'].value_counts())

# Corr plot
from sklearn.decomposition import PCA
F_GPT = new_feature[["histologic_grade", "resection_margin", "lymphovascular_invasion_present", "perineural_invasion_present", "extracapsular_extension", "pathologic_T", "pathologic_N", "pathologic_stage", "localization", "HPV", "metastatic_lymph_node_ratio"]]
M = meta[["age", "gender", "alcohol", "smoking", "pathologic_M", "postoperative_rx_tx"]]
A = new_feature[["extranodal_extension", "histotype"]]

X = pd.concat((M, F_GPT, A), axis=1)

num_pc = 10
pca = PCA(n_components=num_pc)
embeddings_transformed = pca.fit_transform(embeddings)

X = pd.concat((X, pd.DataFrame(embeddings_transformed, columns=["embedding_pc"+str(i) for i in range(num_pc)])), axis=1).dropna()
sns.set_theme(style="white")
corr = X.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(30, 30))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.tight_layout()
plt.savefig("fig/corr_all.png")
plt.close()


# # Embedding plot
# emb = pd.DataFrame(embeddings_transformed, columns=["embedding_pc"+str(i) for i in range(num_pc)])

# features1 = ['age', 'extracapsular_extension', 'metastatic_lymph_node_ratio', 'pathologic_N', 'pathologic_T', 'pathologic_stage', 'perineural_invasion_present', 'postoperative_rx_tx', 'resection_margin']
# features2 = ['age', 'embedding_pc3', 'embedding_pc4', 'embedding_pc8', 'extracapsular_extension', 'metastatic_lymph_node_ratio', 'pathologic_N', 'pathologic_T', 'pathologic_stage', 'perineural_invasion_present', 'postoperative_rx_tx', 'resection_margin']

# X0 = emb[['embedding_pc3', 'embedding_pc4', 'embedding_pc8']]
# data0 = pd.concat((emb, meta[['survival']]), axis=1).dropna().reset_index(drop=True)
# X1 = pd.concat((M, F_GPT, A), axis=1)[features1]
# data1 = pd.concat((X1, meta[['survival']]), axis=1).dropna().reset_index(drop=True)
# X2 = pd.concat((M, F_GPT, A, emb), axis=1)[features2]
# data2 = pd.concat((X2, meta[['survival']]), axis=1).dropna().reset_index(drop=True)

# pca = PCA(n_components=2)
# data_transformed = pca.fit_transform(data[features].to_numpy())

# import umap
# reducer = umap.UMAP(n_neighbors=5)
# data_transformed = reducer.fit_transform(data0[X0.columns].to_numpy())
# umaps = pd.DataFrame(data_transformed, columns=["umap1","umap2"])
# umaps['survival'] = data0.survival

# sns.scatterplot(data=umaps, x="umap1", y="umap2", hue="survival")
# plt.savefig("fig/umap_feature0.png")
# plt.close()

# reducer = umap.UMAP(n_neighbors=15)
# data_transformed = reducer.fit_transform(data1[X1.columns].to_numpy())
# umaps = pd.DataFrame(data_transformed, columns=["umap1","umap2"])
# umaps['survival'] = data1.survival

# sns.scatterplot(data=umaps, x="umap1", y="umap2", hue="survival")
# plt.savefig("fig/umap_feature1.png")
# plt.close()

# reducer = umap.UMAP(n_neighbors=15)
# data_transformed = reducer.fit_transform(data2[X2.columns].to_numpy())
# umaps = pd.DataFrame(data_transformed, columns=["umap1","umap2"])
# umaps['survival'] = data2.survival

# sns.scatterplot(data=umaps, x="umap1", y="umap2", hue="survival")
# plt.savefig("fig/umap_feature2.png")
# plt.close()


# from sklearn.cluster import KMeans
# from lifelines import KaplanMeierFitter
# from lifelines.statistics import logrank_test, multivariate_logrank_test

# target = "OS"
# X = pd.concat((M, F_GPT, A, emb), axis=1)[features2]
# data = pd.concat((X, meta[[target, target+"_censor"]]), axis=1).dropna().reset_index(drop=True)
# kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(data[features2])
# cluster = kmeans.predict(data[features2])
# data['cluster'] = cluster

# reducer = umap.UMAP(n_neighbors=15)
# data_transformed = reducer.fit_transform(data[features2].to_numpy())
# umaps = pd.DataFrame(data_transformed, columns=["umap1","umap2"])
# umaps['cluster'] = data.cluster
# sns.scatterplot(data=umaps, x="umap1", y="umap2", hue="cluster")
# plt.savefig("fig/umap_feature2_cluster.png")
# plt.close()

# kmf = KaplanMeierFitter()
# list_grouped_df = []
# for name, grouped_df in data.groupby('cluster', observed=False):
#     kmf.fit(grouped_df[target], grouped_df[target+"_censor"], label=name)
#     kmf.plot_survival_function()
#     list_grouped_df.append((name, grouped_df))
# # axes[i].set_title(feature, fontsize=20)
# # axes[i].tick_params(axis='x', labelsize=20)
# # axes[i].tick_params(axis='y', labelsize=20)
# # axes[i].set_xlabel('Timeline', fontdict={'fontsize': 20})
# # axes[i].legend(fontsize=15) 

# import itertools
# for group1, group2 in list(itertools.combinations(list_grouped_df, 2)):
#     # results = logrank_test(group1[1][target], group2[1][target],
#     #                     event_observed_A=group1[1][target+"_censor"], event_observed_B=group2[1][target+"_censor"])
    
#     p = results.summary.loc[0, 'p']
#     # 0.0001 0.001 0.01 0.05
#     if p <= 0.0001:
#         print(group1[0], group2[0], p, "****")
#     elif p > 0.0001 and p <= 0.001:
#         print(group1[0], group2[0], p, "***")
#     elif p > 0.001 and p <= 0.01:
#         print(group1[0], group2[0], p, "**")
#     elif p > 0.01 and p <= 0.05:
#         print(group1[0], group2[0], p, "*")
#     else:
#         print(group1[0], group2[0], p, "ns")                    

# results = multivariate_logrank_test

# plt.tight_layout()
# plt.savefig("fig/km_cluster.png")
# plt.close()





# Creating a single frame with multiple plots
fig, axs = plt.subplots(2, 3, figsize=(18, 12))  # Adjust the size as needed

# Age plot
sns.histplot(data=meta, x='age', ax=axs[0, 0])
axs[0, 0].set_title('Age Distribution', fontsize=20)
axs[0, 0].tick_params(axis='x', labelsize=20)
axs[0, 0].tick_params(axis='y', labelsize=20)
axs[0, 0].set_xlabel('', fontsize=20)
axs[0, 0].set_ylabel('Counts', fontdict={'fontsize': 20})

# Gender plot
meta['gender'] = pd.Categorical(meta['gender'].replace({2:'Male', 1:'Female'}), categories=['Female', 'Male'], ordered=True)
sns.histplot(data=meta, x='gender', shrink=.8, ax=axs[0, 1])
axs[0, 1].set_title('Gender Distribution', fontsize=20)
axs[0, 1].tick_params(axis='x', labelsize=20)
axs[0, 1].tick_params(axis='y', labelsize=20)
axs[0, 1].set_xlabel('', fontsize=20)
axs[0, 1].set_ylabel('Counts', fontdict={'fontsize': 20})

# Alcohol plot
meta['alcohol'] = pd.Categorical(meta['alcohol'].replace({2:'Yes', 1:'No', 0:'NA'}), categories=['NA', 'No', 'Yes'], ordered=True)
sns.histplot(data=meta, x='alcohol', shrink=.8, ax=axs[0, 2])
axs[0, 2].set_title('Alcohol Distribution', fontsize=20)
axs[0, 2].tick_params(axis='x', labelsize=20)
axs[0, 2].tick_params(axis='y', labelsize=20)
axs[0, 2].set_xlabel('', fontsize=20)
axs[0, 2].set_ylabel('Counts', fontdict={'fontsize': 20})

# Smoking plot
meta['smoking'] = pd.Categorical(meta['smoking'].replace({3:'Current', 2:'Past', 1:'Never', 0:'NA'}), categories=['NA', 'Never', 'Past', 'Current'], ordered=True)
sns.histplot(data=meta, x='smoking', shrink=.8, ax=axs[1, 0])
axs[1, 0].set_title('Smoking Distribution', fontsize=20)
axs[1, 0].tick_params(axis='x', labelsize=20)
axs[1, 0].tick_params(axis='y', labelsize=20)
axs[1, 0].set_xlabel('', fontsize=20)
axs[1, 0].set_ylabel('Counts', fontdict={'fontsize': 20})

# Pathologic M plot
meta['pathologic_M'] = pd.Categorical(meta['pathologic_M'].replace({2:'M1', 1:'M0 or MX', 0:'NA'}), categories=['NA', 'M0 or MX', 'M1'], ordered=True)
sns.histplot(data=meta, x='pathologic_M', shrink=.8, ax=axs[1, 1])
axs[1, 1].set_title('Pathologic_M Distribution', fontsize=20)
axs[1, 1].tick_params(axis='x', labelsize=20)
axs[1, 1].tick_params(axis='y', labelsize=20)
axs[1, 1].set_xlabel('', fontsize=20)
axs[1, 1].set_ylabel('Counts', fontdict={'fontsize': 20})

# Postoperative Rx Tx plot
meta['postoperative_rx_tx'] = pd.Categorical(meta['postoperative_rx_tx'].replace({2:'Yes', 1:'No', 0:'NA'}), categories=['NA', 'No', 'Yes'], ordered=True)
sns.histplot(data=meta, x='postoperative_rx_tx', shrink=.8, ax=axs[1, 2])
axs[1, 2].set_title('Postoperative_Rx_Tx Distribution', fontsize=20)
axs[1, 2].tick_params(axis='x', labelsize=20)
axs[1, 2].tick_params(axis='y', labelsize=20)
axs[1, 2].set_xlabel('', fontsize=20)
axs[1, 2].set_ylabel('Counts', fontdict={'fontsize': 20})


# Adjust layout
plt.tight_layout()

# Remove the empty subplot (if any)
fig.delaxes(axs[1][2])

plt.savefig("fig/meta_combined.png")
plt.close()






# Combining the plots into one frame
fig, axs = plt.subplots(3, 4, figsize=(25, 15))  # Adjust the layout size as needed
axs = axs.flatten()

for i, feature in enumerate(["histologic_grade", "resection_margin", "lymphovascular_invasion_present","perineural_invasion_present", "extracapsular_extension", 
                             "pathologic_T", "pathologic_N", "pathologic_stage", "localization", "HPV", "metastatic_lymph_node_ratio"]):
    combined_df = pd.DataFrame({
        feature: pd.concat([meta[feature], new_feature[feature]]),
        'data': ['TCGA']*len(meta) + ['GPT-4']*len(new_feature)
    })

    if feature == "histologic_grade":
        combined_df = combined_df.replace({3:'G3',2:'G2',1:'G1',0:'NA'})
        combined_df['histologic_grade'] = pd.Categorical(combined_df['histologic_grade'], categories=['NA', 'G1', 'G2', 'G3'], ordered=True)
    elif feature == "resection_margin":
        combined_df = combined_df.replace({3:'Positive',2:'Close',1:'Clear',0:'NA'})
        combined_df['resection_margin'] = pd.Categorical(combined_df['resection_margin'], categories=['NA', 'Clear', 'Close', 'Positive'], ordered=True)
    elif feature == "lymphovascular_invasion_present":
        combined_df = combined_df.replace({2:'Yes',1:'No',0:'NA'})
        combined_df['lymphovascular_invasion_present'] = pd.Categorical(combined_df['lymphovascular_invasion_present'], categories=['NA', 'No', 'Yes'], ordered=True)
    elif feature == "perineural_invasion_present":
        combined_df = combined_df.replace({2:'Yes',1:'No',0:'NA'})
        combined_df['perineural_invasion_present'] = pd.Categorical(combined_df['perineural_invasion_present'], categories=['NA', 'No', 'Yes'], ordered=True)
    elif feature == "extracapsular_extension":
        combined_df = combined_df.replace({3:'Gross\nextension',2:'Microscopic\nextension',1:'No extracapsular\nextension',0:'NA'})
        combined_df['extracapsular_extension'] = pd.Categorical(combined_df['extracapsular_extension'], categories=['NA', 'No extracapsular\nextension', 'Microscopic\nextension', 'Gross\nextension'], ordered=True)
    elif feature == "pathologic_T":
        combined_df = combined_df.replace({6:'T4b',5:'T4a',4:'T3',3:'T2',2:'T1',1:'T0',0:'TX or NA'})
        combined_df['pathologic_T'] = pd.Categorical(combined_df['pathologic_T'], categories=['TX or NA', 'T0', 'T1', 'T2', 'T3', 'T4a', 'T4b'], ordered=True)
    elif feature == "pathologic_N":
        combined_df = combined_df.replace({4:'N3',3:'N2',2:'N1',1:'N0',0:'NX or NA'})
        combined_df['pathologic_N'] = pd.Categorical(combined_df['pathologic_N'], categories=['NX or NA', 'N0', 'N1', 'N2', 'N3'], ordered=True)
    elif feature == "pathologic_stage":
        combined_df = combined_df.replace({4:'Stage IV',3:'Stage III',2:'Stage II',1:'Stage I',0:'NA'})
        combined_df['pathologic_stage'] = pd.Categorical(combined_df['pathologic_stage'], categories=['NA', 'Stage I', 'Stage II', 'Stage III', 'Stage IV'], ordered=True)
    elif feature == "localization":
        combined_df = combined_df.replace({4:'Hypopharynx',3:'Larynx',2:'Oropharynx',1:'Oral\ncavity',0:'NA'})
        combined_df['localization'] = pd.Categorical(combined_df['localization'], categories=['NA', 'Oral\ncavity', 'Oropharynx', 'Larynx', 'Hypopharynx'], ordered=True)
    elif feature == "HPV":
        combined_df = combined_df.replace({2:'Positive',1:'Negative',0:'NA'})
        combined_df['HPV'] = pd.Categorical(combined_df['HPV'], categories=['NA', 'Negative', 'Positive'], ordered=True)

    if feature == "metastatic_lymph_node_ratio":
        # Metastatic Lymph Node Ratio plot
        sns.histplot(data=combined_df.fillna(0), x=feature, hue='data', ax=axs[i])
        axs[i].set_title(f'{feature} distribution', fontsize=18)
        axs[i].tick_params(axis='x', labelsize=13)
        axs[i].tick_params(axis='y', labelsize=20)
        axs[i].set_xlabel('', fontsize=20)
        axs[i].set_ylabel('Counts', fontdict={'fontsize': 20})
    elif feature == "extracapsular_extension":
        sns.histplot(data=combined_df, x=feature, hue='data', multiple='dodge', shrink=.8, ax=axs[i])
        axs[i].set_title(f'{feature} distribution', fontsize=18)
        axs[i].tick_params(axis='x', labelsize=10)
        axs[i].tick_params(axis='y', labelsize=20)
        axs[i].set_xlabel('', fontsize=20)
        axs[i].set_ylabel('Counts', fontdict={'fontsize': 20})
    else:
        sns.histplot(data=combined_df, x=feature, hue='data', multiple='dodge', shrink=.8, ax=axs[i])
        axs[i].set_title(f'{feature} distribution', fontsize=18)
        axs[i].tick_params(axis='x', labelsize=13)
        axs[i].tick_params(axis='y', labelsize=20)
        axs[i].set_xlabel('', fontsize=20)
        axs[i].set_ylabel('Counts', fontdict={'fontsize': 20})

plt.tight_layout()

# Remove the empty subplot (if any)
axs[11].axis('off')  # This effectively "hides" the axis at position (2, 3)

plt.savefig("fig/meta_vs_newfeature_combined.png")
plt.close()



# Creating a single frame with multiple plots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Adjust the size as needed

# extranodal_extension plot
new_feature['extranodal_extension'] = pd.Categorical(new_feature['extranodal_extension'].replace({2:'Yes', 1:'No', 0:'NA'}), categories=['NA', 'No', 'Yes'], ordered=True)
sns.histplot(data=new_feature, x='extranodal_extension', color=sns.color_palette()[1], shrink=.8, ax=axs[0])
axs[0].set_title('Extranodal Extension Distribution', fontsize=20)
axs[0].tick_params(axis='x', labelsize=20)
axs[0].tick_params(axis='y', labelsize=20)
axs[0].set_xlabel('', fontsize=20)
axs[0].set_ylabel('Counts', fontdict={'fontsize': 20})

# Gender plot
new_feature['histotype'] = pd.Categorical(new_feature['histotype'].replace({2:'Keratinizing', 1:'Basaloid', 0:'NA'}), categories=['NA', 'Basaloid', 'Keratinizing'], ordered=True)
sns.histplot(data=new_feature, x='histotype', color=sns.color_palette()[1], shrink=.8, ax=axs[1])
axs[1].set_title('Histotype Distribution', fontsize=20)
axs[1].tick_params(axis='x', labelsize=20)
axs[1].tick_params(axis='y', labelsize=20)
axs[1].set_xlabel('', fontsize=20)
axs[1].set_ylabel('Counts', fontdict={'fontsize': 20})

# Adjust layout
plt.tight_layout()

plt.savefig("fig/newfeature_combined.png")
plt.close()


from sklearn.metrics import accuracy_score, confusion_matrix
for i, feature in enumerate(["histologic_grade", "resection_margin", "lymphovascular_invasion_present","perineural_invasion_present", "extracapsular_extension", 
                             "pathologic_T", "pathologic_N", "pathologic_stage", "localization", "HPV"]):
    combined_df = pd.concat((meta[feature], new_feature[feature]), axis=1)
    combined_df.columns = ['meta', 'new_feature']
    print(feature)
    print((combined_df['meta']==0).sum())
    combined_df = combined_df[combined_df['meta'] != 0]

    print((combined_df['meta'] == combined_df['new_feature']).sum(), len(combined_df))

    print(accuracy_score(combined_df['meta'], combined_df['new_feature']))
    print(confusion_matrix(combined_df['meta'], combined_df['new_feature']))

# metastatic_lymph_node_ratio
new_feature['metastatic_lymph_node_ratio'] = pd.cut(new_feature['metastatic_lymph_node_ratio'], bins=3, labels=['0th-33th', '33th-66th', '66th-100th'], include_lowest=True)
meta['metastatic_lymph_node_ratio'] = pd.cut(meta['metastatic_lymph_node_ratio'], bins=3, labels=['0th-33th', '33th-66th', '66th-100th'], include_lowest=True)
combined_df = pd.concat((meta['metastatic_lymph_node_ratio'], new_feature['metastatic_lymph_node_ratio']), axis=1)
combined_df.columns = ['meta', 'new_feature']
print('metastatic_lymph_node_ratio')
print((combined_df['meta'].isna()).sum())
combined_df = combined_df.dropna()

print((combined_df['meta'] == combined_df['new_feature']).sum(), len(combined_df))

print(accuracy_score(combined_df['meta'], combined_df['new_feature']))
print(confusion_matrix(combined_df['meta'], combined_df['new_feature']))



