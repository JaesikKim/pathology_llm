import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sns.set_theme(style="white")

OS = pd.DataFrame(
    [['F_base',0.5576385002797986],
     ['F_base',0.6184098639455783],
     ['F_base',0.6475442834138486],
     ['F_base',0.6163973668461998],
     ['F_base',0.6098784997358689],
     ['F_base + F_TCGA',0.5844991606043649],
     ['F_base + F_TCGA',0.642219387755102],
     ['F_base + F_TCGA',0.605475040257649],
     ['F_base + F_TCGA',0.6316576900059845],
     ['F_base + F_TCGA',0.6296883254094031],
     ['F_base + F_GPT',0.6751538891997761],
     ['F_base + F_GPT',0.7380952380952381],
     ['F_base + F_GPT',0.6606280193236715],
     ['F_base + F_GPT',0.6454219030520646],
     ['F_base + F_GPT',0.6397253037506603],
     ['F_base + F_GPT + F_add',0.6773922775601567],
     ['F_base + F_GPT + F_add',0.7346938775510204],
     ['F_base + F_GPT + F_add',0.6598228663446055],
     ['F_base + F_GPT + F_add',0.6466187911430281],
     ['F_base + F_GPT + F_add',0.6339144215530903],
     ['F_base + F_GPT + E_PR',0.7022943480693901],
     ['F_base + F_GPT + E_PR',0.7380952380952381],
     ['F_base + F_GPT + E_PR',0.6892109500805152],
     ['F_base + F_GPT + E_PR',0.6558946738479952],
     ['F_base + F_GPT + E_PR',0.6693079767564712]
    ],
    columns=['model', 'C-index']
)

DSS = pd.DataFrame(
    [['F_base',0.5839630562552477],
     ['F_base',0.5950155763239875],
     ['F_base',0.6838180462341537],
     ['F_base',0.6791590493601463],
     ['F_base',0.572618254497002],
     ['F_base + F_TCGA',0.6276238455079765],
     ['F_base + F_TCGA',0.6196261682242991],
     ['F_base + F_TCGA',0.639075316927666],
     ['F_base + F_TCGA',0.7285191956124314],
     ['F_base + F_TCGA',0.6645569620253164],
     ['F_base + F_GPT',0.7527287993282955],
     ['F_base + F_GPT',0.7003115264797508],
     ['F_base + F_GPT',0.6662938105891126],
     ['F_base + F_GPT',0.8162705667276051],
     ['F_base + F_GPT',0.7258494337108594],
     ['F_base + F_GPT + F_add',0.7493702770780857],
     ['F_base + F_GPT + F_add',0.6934579439252336],
     ['F_base + F_GPT + F_add',0.6662938105891126],
     ['F_base + F_GPT + F_add',0.8016453382084096],
     ['F_base + F_GPT + F_add',0.7245169886742172],
     ['F_base + F_GPT + E_PR',0.7732997481108312],
     ['F_base + F_GPT + E_PR',0.7102803738317757],
     ['F_base + F_GPT + E_PR',0.6703952274422073],
     ['F_base + F_GPT + E_PR',0.8135283363802559],
     ['F_base + F_GPT + E_PR',0.7514990006662225]
    ],
    columns=['model', 'C-index']
)

PFS = pd.DataFrame(
    [['F_base',0.5],
     ['F_base',0.5],
     ['F_base',0.5],
     ['F_base',0.5],
     ['F_base',0.5],
     ['F_base + F_TCGA',0.6194556451612904],
     ['F_base + F_TCGA',0.5881183745583038],
     ['F_base + F_TCGA',0.5458715596330275],
     ['F_base + F_TCGA',0.5149456521739131],
     ['F_base + F_TCGA',0.6401826484018265],
     ['F_base + F_GPT',0.657258064516129],
     ['F_base + F_GPT',0.6601148409893993],
     ['F_base + F_GPT',0.5751146788990825],
     ['F_base + F_GPT',0.5292119565217391],
     ['F_base + F_GPT',0.7036529680365297],
     ['F_base + F_GPT + F_add',0.6582661290322581],
     ['F_base + F_GPT + F_add',0.6587897526501767],
     ['F_base + F_GPT + F_add',0.5762614678899083],
     ['F_base + F_GPT + F_add',0.5285326086956522],
     ['F_base + F_GPT + F_add',0.7027397260273973],
     ['F_base + F_GPT + E_PR',0.6391129032258065],
     ['F_base + F_GPT + E_PR',0.6601148409893993],
     ['F_base + F_GPT + E_PR',0.573394495412844],
     ['F_base + F_GPT + E_PR',0.5360054347826086],
     ['F_base + F_GPT + E_PR',0.6970319634703196]
    ],
    columns=['model', 'C-index']
)

fig, ax = plt.subplots(figsize=(6,6))
sns.barplot(OS, x='model', y='C-index', errorbar="sd", palette=sns.color_palette())
ax.set(ylim=(0.5,0.82))
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_xlabel('')
ax.set_ylabel('C-index', fontdict={'fontsize': 20})

plt.tight_layout()
plt.savefig("fig/fig3_1.png")
plt.close()

fig, ax = plt.subplots(figsize=(6,6))
sns.barplot(DSS, x='model', y='C-index', errorbar="sd", palette=sns.color_palette())
ax.set(ylim=(0.5,0.82))
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_xlabel('')
ax.set_ylabel('C-index', fontdict={'fontsize': 20})

plt.tight_layout()
plt.savefig("fig/fig3_2.png")
plt.close()

# fig, ax = plt.subplots(figsize=(6,6))
# sns.barplot(PFS, x='model', y='C-index', errorbar="sd", palette=sns.color_palette())
# ax.set(ylim=(0.5,0.82))
# ax.tick_params(axis='x', labelsize=20)
# ax.tick_params(axis='y', labelsize=20)
# ax.set_xticks([])
# ax.set_xticklabels([])
# ax.set_xlabel('')
# ax.set_ylabel('C-index', fontdict={'fontsize': 20})

# plt.tight_layout()
# plt.savefig("fig/fig3_3.png")
# plt.close()


OS_selected_features = [['age', 'pathologic_M', 'postoperative_rx_tx', 'resection_margin', 'perineural_invasion_present', 'extracapsular_extension', 'pathologic_T', 'pathologic_N', 'pathologic_stage', 'metastatic_lymph_node_ratio', 'embedding_pc3', 'embedding_pc4', 'embedding_pc8'], ['age', 'postoperative_rx_tx', 'resection_margin', 'perineural_invasion_present', 'extracapsular_extension', 'pathologic_T', 'pathologic_N', 'pathologic_stage', 'metastatic_lymph_node_ratio', 'embedding_pc4', 'embedding_pc7'], ['age', 'postoperative_rx_tx', 'resection_margin', 'perineural_invasion_present', 'extracapsular_extension', 'pathologic_T', 'pathologic_N', 'pathologic_stage', 'metastatic_lymph_node_ratio', 'embedding_pc3', 'embedding_pc4', 'embedding_pc8'], ['age', 'postoperative_rx_tx', 'resection_margin', 'perineural_invasion_present', 'extracapsular_extension', 'pathologic_T', 'pathologic_N', 'pathologic_stage', 'metastatic_lymph_node_ratio', 'embedding_pc3', 'embedding_pc4', 'embedding_pc7'], ['age', 'postoperative_rx_tx', 'resection_margin', 'perineural_invasion_present', 'extracapsular_extension', 'pathologic_T', 'pathologic_N', 'pathologic_stage', 'metastatic_lymph_node_ratio', 'embedding_pc3', 'embedding_pc8']]

DSS_selected_features = [['pathologic_M', 'postoperative_rx_tx', 'resection_margin', 'perineural_invasion_present', 'extracapsular_extension', 'pathologic_T', 'pathologic_N', 'pathologic_stage', 'metastatic_lymph_node_ratio', 'embedding_pc4', 'embedding_pc5'], ['postoperative_rx_tx', 'resection_margin', 'lymphovascular_invasion_present', 'perineural_invasion_present', 'extracapsular_extension', 'pathologic_N', 'metastatic_lymph_node_ratio', 'embedding_pc4', 'embedding_pc8'], ['postoperative_rx_tx', 'resection_margin', 'perineural_invasion_present', 'extracapsular_extension', 'pathologic_T', 'pathologic_N', 'pathologic_stage', 'metastatic_lymph_node_ratio', 'embedding_pc3', 'embedding_pc4', 'embedding_pc8'], ['postoperative_rx_tx', 'resection_margin', 'perineural_invasion_present', 'extracapsular_extension', 'pathologic_T', 'pathologic_N', 'pathologic_stage', 'metastatic_lymph_node_ratio', 'embedding_pc4'], ['postoperative_rx_tx', 'resection_margin', 'perineural_invasion_present', 'extracapsular_extension', 'pathologic_T', 'pathologic_N', 'pathologic_stage', 'metastatic_lymph_node_ratio', 'embedding_pc4', 'embedding_pc8']]

# PFS_selected_features = [['resection_margin', 'extracapsular_extension', 'pathologic_N', 'metastatic_lymph_node_ratio', 'embedding_pc0', 'embedding_pc4'], ['resection_margin', 'extracapsular_extension', 'pathologic_N', 'metastatic_lymph_node_ratio'], ['resection_margin', 'perineural_invasion_present', 'extracapsular_extension', 'pathologic_T', 'pathologic_N', 'pathologic_stage', 'metastatic_lymph_node_ratio', 'localization_4.0', 'embedding_pc4', 'embedding_pc8'], ['resection_margin', 'extracapsular_extension', 'pathologic_N', 'metastatic_lymph_node_ratio', 'embedding_pc4'], ['extracapsular_extension', 'pathologic_N', 'metastatic_lymph_node_ratio']]

# Combine all features and create a set for unique features
all_features = sorted(set(feature for features_list in (OS_selected_features + DSS_selected_features) for feature in features_list))

# Create an empty DataFrame with all features as index
df = pd.DataFrame(index=all_features)

# Populate the DataFrame for OS, DSS, PFS
for idx, category in enumerate([OS_selected_features, DSS_selected_features], start=1):
    for experiment, features in enumerate(category):
        col_name = f'{"OS" if idx == 1 else "DSS"} - fold {experiment+1}'
        df[col_name] = 0
        df.loc[features, col_name] = 1

# Define color maps for each category
os_cmap = ListedColormap(['lightgrey', 'skyblue'])
# dss_cmap = ListedColormap(['lightgrey', 'lightgreen'])
dss_cmap = ListedColormap(['lightgrey', 'salmon'])

# Plot heatmaps for each category
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8.5, 6), sharey=True)

# OS Heatmap
sns.heatmap(df[[col for col in df if "OS" in col]], ax=axes[0], cmap=os_cmap, linewidths=.5, cbar=False)
axes[0].set_title('OS Selected Features')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")


# DSS Heatmap
sns.heatmap(df[[col for col in df if "DSS" in col]], ax=axes[1], cmap=dss_cmap, linewidths=.5, cbar=False)
axes[1].set_title('DSS Selected Features')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")

# # PFS Heatmap
# sns.heatmap(df[[col for col in df if "PFS" in col]], ax=axes[2], cmap=pfs_cmap, linewidths=.5, cbar=False)
# axes[2].set_title('PFS Selected Features')
# axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig("fig/fig3_4.png")
plt.close()










OS = pd.DataFrame(
    [[r'$\boldsymbol{E_{Raw}}$',0.5204252937884724],
     [r'$\boldsymbol{E_{Raw}}$',0.45918367346938777],
     [r'$\boldsymbol{E_{Raw}}$',0.5],
     [r'$\boldsymbol{E_{Raw}}$',0.48],
     [r'$\boldsymbol{E_{Raw}}$',0.5295416896742131],
     [r'$\boldsymbol{E_{PR}}$',0.604364857302742],
     [r'$\boldsymbol{E_{PR}}$',0.6305272108843537],
     [r'$\boldsymbol{E_{PR}}$',0.6763285024154589],
     [r'$\boldsymbol{E_{PR}}$',0.6533333333333333],
     [r'$\boldsymbol{E_{PR}}$',0.6598564329099945],
    #  ['GPT extraction',0.6583031557165029],
    #  ['GPT extraction',0.7076784912438258],
    #  ['GPT extraction',0.6494802494802495],
    #  ['GPT extraction',0.5893060295790671],
    #  ['GPT extraction',0.6590204587724736],
    #  ['GPT extraction\n+Raw text embeddings (10 PCs)',0.6595964821520952],
    #  ['GPT extraction\n+Raw text embeddings (10 PCs)',0.7099236641221374],
    #  ['GPT extraction\n+Raw text embeddings (10 PCs)',0.6577962577962578],
    #  ['GPT extraction\n+Raw text embeddings (10 PCs)',0.590443686006826],
    #  ['GPT extraction\n+Raw text embeddings (10 PCs)',0.6478611283323],
    #  ['GPT extraction\n+PR embddings (10 PCs)',0.6828763579927574],
    #  ['GPT extraction\n+PR embddings (10 PCs)',0.7036371800628648],
    #  ['GPT extraction\n+PR embddings (10 PCs)',0.6432432432432432],
    #  ['GPT extraction\n+PR embddings (10 PCs)',0.6131968145620023],
    #  ['GPT extraction\n+PR embddings (10 PCs)',0.6652200867947923],
    ],
    columns=['Model', 'C-index']
)

DSS = pd.DataFrame(
    [[r'$\boldsymbol{E_{Raw}}$',0.5079764903442485],
     [r'$\boldsymbol{E_{Raw}}$',0.45981308411214955],
     [r'$\boldsymbol{E_{Raw}}$',0.5],
     [r'$\boldsymbol{E_{Raw}}$',0.5],
     [r'$\boldsymbol{E_{Raw}}$',0.5191146881287726],
     [r'$\boldsymbol{E_{PR}}$',0.6246851385390428],
     [r'$\boldsymbol{E_{PR}}$',0.6772585669781932],
     [r'$\boldsymbol{E_{PR}}$',0.6002982848620433],
     [r'$\boldsymbol{E_{PR}}$',0.7680311890838206],
     [r'$\boldsymbol{E_{PR}}$',0.6626425217974514],
    #  ['GPT extraction',0.7476948868398994],
    #  ['GPT extraction',0.7097355769230769],
    #  ['GPT extraction',0.6478300180831826],
    #  ['GPT extraction',0.7266099635479951],
    #  ['GPT extraction',0.7876506024096386],
    #  ['GPT extraction\n+Raw text embeddings (10 PCs)',0.7367979882648784],
    #  ['GPT extraction\n+Raw text embeddings (10 PCs)',0.7016225961538461],
    #  ['GPT extraction\n+Raw text embeddings (10 PCs)',0.6735985533453888],
    #  ['GPT extraction\n+Raw text embeddings (10 PCs)',0.7095990279465371],
    #  ['GPT extraction\n+Raw text embeddings (10 PCs)',0.7786144578313253],
    #  ['GPT extraction\n+PR embddings (10 PCs)',0.7770326906957251],
    #  ['GPT extraction\n+PR embddings (10 PCs)',0.7091346153846154],
    #  ['GPT extraction\n+PR embddings (10 PCs)',0.635623869801085],
    #  ['GPT extraction\n+PR embddings (10 PCs)',0.7120291616038882],
    #  ['GPT extraction\n+PR embddings (10 PCs)',0.7801204819277109],
    ],
    columns=['Model', 'C-index']
)

# PFS = pd.DataFrame(
#     [[r'$\boldsymbol{E_{Raw}}$',0.5],
#      [r'$\boldsymbol{E_{Raw}}$',0.5],
#      [r'$\boldsymbol{E_{Raw}}$',0.5],
#      [r'$\boldsymbol{E_{Raw}}$',0.5],
#      [r'$\boldsymbol{E_{Raw}}$',0.5],
#      [r'$\boldsymbol{E_{PR}}$',0.5443548387096774],
#      [r'$\boldsymbol{E_{PR}}$',0.5],
#      [r'$\boldsymbol{E_{PR}}$',0.4868119266055046],
#      [r'$\boldsymbol{E_{PR}}$',0.5],
#      [r'$\boldsymbol{E_{PR}}$',0.5],
#     #  ['GPT extraction',0.6269193857965452],
#     #  ['GPT extraction',0.6048613901165126],
#     #  ['GPT extraction',0.6160092807424594],
#     #  ['GPT extraction',0.4924607961399276],
#     #  ['GPT extraction',0.7000467945718297],
#     #  ['GPT extraction\n+Raw text embeddings (10 PCs)',0.652831094049904],
#     #  ['GPT extraction\n+Raw text embeddings (10 PCs)',0.6139011651265569],
#     #  ['GPT extraction\n+Raw text embeddings (10 PCs)',0.5784996133023975],
#     #  ['GPT extraction\n+Raw text embeddings (10 PCs)',0.4927623642943305],
#     #  ['GPT extraction\n+Raw text embeddings (10 PCs)',0.7063640617688348],
#     #  ['GPT extraction\n+PR embddings (10 PCs)',0.6156429942418427],
#     #  ['GPT extraction\n+PR embddings (10 PCs)',0.6139011651265569],
#     #  ['GPT extraction\n+PR embddings (10 PCs)',0.576952822892498],
#     #  ['GPT extraction\n+PR embddings (10 PCs)',0.4927623642943305],
#     #  ['GPT extraction\n+PR embddings (10 PCs)',0.7063640617688348],
#     ],
#     columns=['Model', 'C-index']
# )


fig, axes = plt.subplots(1,2, figsize=(10,6))

sns.barplot(OS, x='Model', y='C-index', errorbar="sd", palette=[sns.color_palette()[0], sns.color_palette()[1]], ax=axes[0])
axes[0].set(ylim=(0.4,0.75))
axes[0].tick_params(axis='x', labelsize=20)
axes[0].tick_params(axis='y', labelsize=20)
axes[0].set_xlabel('Model', fontdict={'fontsize': 20})
axes[0].set_ylabel('C-index', fontdict={'fontsize': 20})
axes[0].set_title('Overall Survival', fontdict={'fontsize': 20})

sns.barplot(DSS, x='Model', y='C-index', errorbar="sd", palette=[sns.color_palette()[0], sns.color_palette()[1]], ax=axes[1])
axes[1].set(ylim=(0.4,0.75))
axes[1].tick_params(axis='x', labelsize=20)
axes[1].tick_params(axis='y', labelsize=20)
axes[1].set_xlabel('Model', fontdict={'fontsize': 20})
axes[1].set_ylabel('C-index', fontdict={'fontsize': 20})
axes[1].set_title('Disease Specific Survival', fontdict={'fontsize': 20})

# sns.barplot(PFS, x='Model', y='C-index', errorbar="sd", palette=[sns.color_palette()[0], sns.color_palette()[1]], ax=axes[2])
# axes[2].set(ylim=(0.4,0.75))
# axes[2].tick_params(axis='x', labelsize=20)
# axes[2].tick_params(axis='y', labelsize=20)
# axes[2].set_xlabel('Model', fontdict={'fontsize': 20})
# axes[2].set_ylabel('C-index', fontdict={'fontsize': 20})
# # axes[2].set_title('Progression Free Survival', fontdict={'fontsize': 25})

# plt.legend("", fontsize=20)
# plt.ylabel("", fontsize=16)
# plt.show()
plt.tight_layout()
plt.savefig("fig/s_fig4.png")
plt.close()