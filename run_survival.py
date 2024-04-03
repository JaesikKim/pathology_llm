# Survival Analysis with C-index
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import false_discovery_control

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
    if (len(meta_ret) >0) and (len(_data[id]["newfeature_prompt"][0]) == 13): # and (len(_data[id]["embed_raw"]) == 3072): # 1536 // 3072
        meta_idx.append(meta_ret.index[0])
        new_feature.append(_data[id]["newfeature_prompt"][0])
        # embeddings.append(_data[id]["embed_raw"])
        embeddings.append(_data[id]["embed_prompt"][0])
meta = meta.loc[meta_idx].reset_index(drop=True)
print(meta.shape)
embeddings = pd.DataFrame(embeddings)

new_feature = pd.DataFrame(new_feature, columns=["histologic_grade", "resection_margin", "lymphovascular_invasion_present","perineural_invasion_present", "extracapsular_extension", 
                                                 "pathologic_T", "pathologic_N", "pathologic_stage",  
                                                 "extranodal_extension", "localization", "histotype", "HPV", "metastatic_lymph_node_ratio"])
# new_feature['pathologic_TX'] = (new_feature['pathologic_T'] == 1).astype(int)
# new_feature['pathologic_NX'] = (new_feature['pathologic_N'] == 1).astype(int)
new_feature['pathologic_T'] = new_feature['pathologic_T'].replace({1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6})
new_feature['pathologic_N'] = new_feature['pathologic_N'].replace({1:0, 2:1, 3:2, 4:3, 5:4})
meta['metastatic_lymph_node_ratio'] = meta['metastatic_lymph_node_ratio'].fillna(0)

# # test imputation - use new feature and meta
# for target in ["OS", "DSS", "PFS"]:
#     cindexs = []
#     selected_features = []
#     for r in range(5):
#         _new_feature = new_feature[~pd.isna(meta[target])].reset_index(drop=True)
#         _meta = meta[~pd.isna(meta[target])].reset_index(drop=True)
#         _new_feature = _new_feature[~pd.isna(_meta[target+"_censor"])].reset_index(drop=True)
#         _meta = _meta[~pd.isna(_meta[target+"_censor"])].reset_index(drop=True)
        
#         F_TCGA = _meta[["histologic_grade", "resection_margin", "lymphovascular_invasion_present", "perineural_invasion_present", "extracapsular_extension", "pathologic_T", "pathologic_N", "pathologic_stage", "localization", "HPV", "metastatic_lymph_node_ratio"]]
#         F_GPT = _new_feature[["histologic_grade", "resection_margin", "lymphovascular_invasion_present", "perineural_invasion_present", "extracapsular_extension", "pathologic_T", "pathologic_N", "pathologic_stage", "localization", "HPV", "metastatic_lymph_node_ratio"]]
#         tmp = pd.concat((F_TCGA, F_GPT))
#         tmp = pd.get_dummies(tmp, columns=["localization"], dtype=int)
#         tmp = tmp.drop("localization_0.0", axis=1)
#         # tmp = tmp.rename({"localization_0.0":"localization_missing"}, axis=1)
#         F_TCGA = tmp.iloc[:len(F_TCGA),:]
#         F_GPT = tmp.iloc[len(F_TCGA):,:]

#         M = _meta[["age", "gender", "alcohol", "smoking", "pathologic_M", "postoperative_rx_tx"]]
#         A = _new_feature[["extranodal_extension", "histotype"]]

#         # X = M
#         # X = pd.concat((M, F_TCGA), axis=1)
#         X = pd.concat((M, F_GPT), axis=1)

#         X_train = X.loc[_meta['split_'+str(r)] == "train"].reset_index(drop=True)
#         y_train = _meta.loc[_meta['split_'+str(r)] == "train", [target, target+"_censor"]].reset_index(drop=True)
#         X_test = X.loc[_meta['split_'+str(r)] == "test"].reset_index(drop=True)
#         y_test = _meta.loc[_meta['split_'+str(r)] == "test", [target, target+"_censor"]].reset_index(drop=True)
#         print(X.shape, X_train.shape, X_test.shape)

#         from sklearn.impute import KNNImputer
#         impute_columns = X.columns.tolist()
#         impute_true = ~np.isin(impute_columns, ['localization_1.0', 'localization_2.0', 'localization_3.0', 'localization_4.0'])
#         impute_false = np.isin(impute_columns, ['localization_1.0', 'localization_2.0', 'localization_3.0', 'localization_4.0'])
        
#         imputer = KNNImputer(n_neighbors=3, weights="uniform")
#         imputer.fit(X_train.iloc[:,impute_true].replace({0:np.nan}))

#         X_train = pd.concat((pd.DataFrame(imputer.transform(X_train.iloc[:,impute_true].replace({0:np.nan})), columns=X_train.iloc[:,impute_true].columns), X_train.iloc[:,impute_false]), axis=1)
#         X_test = pd.concat((pd.DataFrame(imputer.transform(X_test.iloc[:,impute_true].replace({0:np.nan})), columns=X_test.iloc[:,impute_true].columns), X_test.iloc[:,impute_false]), axis=1)

#         p_values = []
#         for i in range(X_train.shape[1]):
#             if X_train.iloc[:,[i]].var().iloc[0] == 0:
#                 continue
#             df_train = pd.concat((X_train.iloc[:,[i]],y_train), axis=1)
#             cph = CoxPHFitter()
#             cph.fit(df_train, duration_col=target, event_col=target+'_censor')

#             # Print the summary of the model
#             p_values.append([i, cph.summary.p.item()])
#         p_values = np.array(p_values)
#         idx = p_values[false_discovery_control(p_values[:,1]) < 0.05, 0].astype(int)

#         selected_features.append(X_train.columns[idx].tolist())
#         # print("significant columns: ", X_train.columns[idx].tolist())
        

#         df_train = pd.concat((X_train.iloc[:,idx],y_train), axis=1)
#         df_test = pd.concat((X_test.iloc[:,idx],y_test), axis=1)
#         cph = CoxPHFitter()
#         cph.fit(df_train, duration_col=target, event_col=target+'_censor')

#         # df_train = pd.concat((X_train, y_train), axis=1)
#         # df_test = pd.concat((X_test, y_test), axis=1)
#         # cph = CoxPHFitter(penalizer=0.01, l1_ratio=0.5)
#         # cph.fit(df_train, duration_col=target, event_col=target+'_censor')

#         # Compute the C-index
#         c_index = concordance_index(df_test[target], -cph.predict_partial_hazard(df_test), df_test[target+'_censor'])
#         # print("C-index: ", c_index)
#         cindexs.append(c_index)
#     print(target, np.mean(cindexs), np.std(cindexs))
#     print(selected_features)






# # use new feature and meta
# for target in ["OS", "DSS", "PFS"]:
#     cindexs = []
#     selected_features = []
#     for r in range(5):
#         _new_feature = new_feature[~pd.isna(meta[target])].reset_index(drop=True)
#         _meta = meta[~pd.isna(meta[target])].reset_index(drop=True)
#         _new_feature = _new_feature[~pd.isna(_meta[target+"_censor"])].reset_index(drop=True)
#         _meta = _meta[~pd.isna(_meta[target+"_censor"])].reset_index(drop=True)
        
#         F_TCGA = _meta[["histologic_grade", "resection_margin", "lymphovascular_invasion_present", "perineural_invasion_present", "extracapsular_extension", "pathologic_T", "pathologic_N", "pathologic_stage", "localization", "HPV", "metastatic_lymph_node_ratio"]]
#         F_GPT = _new_feature[["histologic_grade", "resection_margin", "lymphovascular_invasion_present", "perineural_invasion_present", "extracapsular_extension", "pathologic_T", "pathologic_N", "pathologic_stage", "localization", "HPV", "metastatic_lymph_node_ratio"]]
#         tmp = pd.concat((F_TCGA, F_GPT))
#         tmp = pd.get_dummies(tmp, columns=["localization"], dtype=int)
#         tmp = tmp.drop("localization_0.0", axis=1)
#         # tmp = tmp.rename({"localization_0.0":"localization_missing"}, axis=1)
#         F_TCGA = tmp.iloc[:len(F_TCGA),:]
#         F_GPT = tmp.iloc[len(F_TCGA):,:]

#         M = _meta[["age", "gender", "alcohol", "smoking", "pathologic_M", "postoperative_rx_tx"]]
#         A = _new_feature[["extranodal_extension", "histotype"]]

#         # X = M
#         # X = pd.concat((M, F_TCGA), axis=1)
#         X = pd.concat((M, F_GPT), axis=1)
#         # X = pd.concat((M, F_GPT, A), axis=1)

#         X_train = X.loc[_meta['split_'+str(r)] == "train"].reset_index(drop=True)
#         y_train = _meta.loc[_meta['split_'+str(r)] == "train", [target, target+"_censor"]].reset_index(drop=True)
#         X_test = X.loc[_meta['split_'+str(r)] == "test"].reset_index(drop=True)
#         y_test = _meta.loc[_meta['split_'+str(r)] == "test", [target, target+"_censor"]].reset_index(drop=True)
#         print(X.shape, X_train.shape, X_test.shape)

#         p_values = []
#         for i in range(X_train.shape[1]):
#             if X_train.iloc[:,[i]].var().iloc[0] == 0:
#                 continue
#             df_train = pd.concat((X_train.iloc[:,[i]],y_train), axis=1)
#             cph = CoxPHFitter()
#             cph.fit(df_train, duration_col=target, event_col=target+'_censor')

#             # Print the summary of the model
#             p_values.append([i, cph.summary.p.item()])
#         p_values = np.array(p_values)
#         idx = p_values[false_discovery_control(p_values[:,1]) < 0.05, 0].astype(int)

#         selected_features.append(X_train.columns[idx].tolist())
#         # print("significant columns: ", X_train.columns[idx].tolist())

#         df_train = pd.concat((X_train.iloc[:,idx],y_train), axis=1)
#         df_test = pd.concat((X_test.iloc[:,idx],y_test), axis=1)
#         cph = CoxPHFitter()
#         cph.fit(df_train, duration_col=target, event_col=target+'_censor')

#         # Compute the C-index
#         c_index = concordance_index(df_test[target], -cph.predict_partial_hazard(df_test), df_test[target+'_censor'])
#         # print("C-index: ", c_index)
#         cindexs.append(c_index)
#     print(target, np.mean(cindexs), np.std(cindexs))
#     print(cindexs)
#     print(selected_features)




# # Lasso/Elastic - use new feature and meta
# l1_ratio=1 #1
# for target in ["OS", "DSS", "PFS"]:
#     cindexs = []
#     selected_features = []
#     for r in range(5):
#         _new_feature = new_feature[~pd.isna(meta[target])].reset_index(drop=True)
#         _meta = meta[~pd.isna(meta[target])].reset_index(drop=True)
#         _new_feature = _new_feature[~pd.isna(_meta[target+"_censor"])].reset_index(drop=True)
#         _meta = _meta[~pd.isna(_meta[target+"_censor"])].reset_index(drop=True)
        
#         meta_B = _meta[["histologic_grade", "resection_margin", "lymphovascular_invasion_present", "perineural_invasion_present", "extracapsular_extension", "pathologic_T", "pathologic_N", "overall_stage", "localization", "HPV"]]
#         report_B = _new_feature[["histologic_grade", "resection_margin", "lymphovascular_invasion_present", "perineural_invasion_present", "extracapsular_extension", "pathologic_T", "pathologic_N", "overall_stage", "localization", "HPV"]]
#         tmp_B = pd.concat((meta_B, report_B))
#         tmp_B = pd.get_dummies(tmp_B, columns=["localization"], dtype=int)
#         tmp_B = tmp_B.drop("localization_0.0", axis=1)
#         # tmp_B = tmp_B.rename({"localization_0.0":"localization_missing"}, axis=1)
#         meta_B = tmp_B.iloc[:len(meta_B),:]
#         report_B = tmp_B.iloc[len(meta_B):,:]

#         M = _meta[["age", "gender", "alcohol", "pathologic_M", "postoperative_rx_tx"]]
#         R = _new_feature[["extranodal_extension", "histotype", "metastatic_lymph_node_ratio"]]

#         # X = meta_B
#         # X = report_B
#         # X = pd.concat((M, meta_B), axis=1)
#         # X = pd.concat((M, report_B), axis=1)
#         # X = pd.concat((M, meta_B, R), axis=1)
#         X = pd.concat((M, report_B, R), axis=1)


#         X_train = X.loc[_meta['split_'+str(r)] == "train"].reset_index(drop=True)
#         y_train = _meta.loc[_meta['split_'+str(r)] == "train", [target, target+"_censor"]].reset_index(drop=True)
#         X_test = X.loc[_meta['split_'+str(r)] == "test"].reset_index(drop=True)
#         y_test = _meta.loc[_meta['split_'+str(r)] == "test", [target, target+"_censor"]].reset_index(drop=True)
#         print(X.shape, X_train.shape, X_test.shape)

#         idx = []
#         for i in range(X_train.shape[1]):
#             if X_train.iloc[:,[i]].var().iloc[0] == 0:
#                 continue
#             idx.append(i)

#         df_train = pd.concat((X_train.iloc[:,idx], y_train), axis=1)
#         df_test = pd.concat((X_test.iloc[:,idx], y_test), axis=1)

#         # Assuming X_train, y_train are defined as per your setup
#         from sklearn.model_selection import KFold
#         kf = KFold(n_splits=5)  # Define the number of splits for cross-validation
#         penalizer_values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]  # Example range of penalizer values
#         c_indices = []

#         for penalizer in penalizer_values:
#             temp_c_indices = []
#             for train_index, val_index in kf.split(X_train):
#                 # Splitting the data into training and validation sets
#                 X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
#                 y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

#                 # Training the model
#                 cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
#                 df_train_cv = pd.concat((X_train_cv, y_train_cv), axis=1)
#                 cph.fit(df_train_cv, duration_col=target, event_col=target+'_censor')

#                 # Evaluating the model on the validation set
#                 df_val_cv = pd.concat((X_val_cv, y_val_cv), axis=1)
#                 ci = concordance_index(df_val_cv[target], -cph.predict_partial_hazard(df_val_cv), df_val_cv[target+'_censor'])
#                 temp_c_indices.append(ci)

#             # Averaging the C-index across all folds for the current penalizer value
#             avg_ci = np.mean(temp_c_indices)
#             c_indices.append(avg_ci)

#         # Selecting the penalizer value with the highest average C-index
#         optimal_penalizer = penalizer_values[np.argmax(c_indices)]

#         # Retrain the model with the optimal penalizer on the entire training set
#         print("optimal_penalizer: ", optimal_penalizer)
#         cph_optimal = CoxPHFitter(penalizer=optimal_penalizer, l1_ratio=l1_ratio)
#         df_train = pd.concat((X_train, y_train), axis=1)
#         cph_optimal.fit(df_train, duration_col=target, event_col=target+'_censor')

# # print(sum(cph.summary.coef>0.001))
#         # selected_features.append(X_train.columns[idx].tolist())
#         # print("significant columns: ", X_train.columns[idx].tolist())

#         # Compute the C-index
#         c_index = concordance_index(df_test[target], -cph.predict_partial_hazard(df_test), df_test[target+'_censor'])
#         # print("C-index: ", c_index)
#         cindexs.append(c_index)
#     print(target, np.mean(cindexs), np.std(cindexs))
#     print(selected_features)





# # Raw embedding / Prompt embedding
# for target in ["OS", "DSS", "PFS"]:
#     cindexs = []
#     selected_features = []
#     for r in range(5):
#         _embeddings = embeddings[~pd.isna(meta[target])].reset_index(drop=True)
#         _new_feature = new_feature[~pd.isna(meta[target])].reset_index(drop=True)
#         _meta = meta[~pd.isna(meta[target])].reset_index(drop=True)
#         _embeddings = _embeddings[~pd.isna(_meta[target+"_censor"])].reset_index(drop=True)
#         _new_feature = _new_feature[~pd.isna(_meta[target+"_censor"])].reset_index(drop=True)
#         _meta = _meta[~pd.isna(_meta[target+"_censor"])].reset_index(drop=True)
        
#         embeddings_train = _embeddings.loc[_meta['split_'+str(r)] == "train"].reset_index(drop=True)
#         y_train = _meta.loc[_meta['split_'+str(r)] == "train", [target, target+"_censor"]].reset_index(drop=True)
#         embeddings_test = _embeddings.loc[_meta['split_'+str(r)] == "test"].reset_index(drop=True)
#         y_test = _meta.loc[_meta['split_'+str(r)] == "test", [target, target+"_censor"]].reset_index(drop=True)

#         from sklearn.decomposition import PCA
#         pca = PCA(n_components=10)
#         embeddings_transformed_train = pca.fit_transform(embeddings_train)
#         embeddings_transformed_test = pca.transform(embeddings_test)

#         X_train = pd.DataFrame(embeddings_transformed_train)
#         X_test = pd.DataFrame(embeddings_transformed_test)

#         p_values = []
#         for i in range(X_train.shape[1]):
#             if X_train.iloc[:,[i]].var().iloc[0] == 0:
#                 continue
#             df_train = pd.concat((X_train.iloc[:,[i]],y_train), axis=1)
#             cph = CoxPHFitter()
#             cph.fit(df_train, duration_col=target, event_col=target+'_censor')

#             # Print the summary of the model
#             p_values.append([i, cph.summary.p.item()])
#         p_values = np.array(p_values)
#         idx = p_values[false_discovery_control(p_values[:,1]) < 0.05, 0].astype(int)

#         selected_features.append(X_train.columns[idx].tolist())

#         df_train = pd.concat((X_train.iloc[:,idx],y_train), axis=1).dropna()
#         df_test = pd.concat((X_test.iloc[:,idx],y_test), axis=1).dropna()
#         cph = CoxPHFitter()
#         cph.fit(df_train, duration_col=target, event_col=target+'_censor')

#         # Compute the C-index
#         c_index = concordance_index(df_test[target], -cph.predict_partial_hazard(df_test), df_test[target+'_censor'])
#         print("C-index: ", c_index)
#         cindexs.append(c_index)
#     print(target, np.mean(cindexs), np.std(cindexs))
#     print(selected_features)




# CoxPH - New feature + Raw embedding / Prompt embedding
for target in ["OS", "DSS"]:
    cindexs = []
    selected_features = []
    for r in range(5):
        _embeddings = embeddings[~pd.isna(meta[target])].reset_index(drop=True)
        _new_feature = new_feature[~pd.isna(meta[target])].reset_index(drop=True)
        _meta = meta[~pd.isna(meta[target])].reset_index(drop=True)
        _embeddings = _embeddings[~pd.isna(_meta[target+"_censor"])].reset_index(drop=True)
        _new_feature = _new_feature[~pd.isna(_meta[target+"_censor"])].reset_index(drop=True)
        _meta = _meta[~pd.isna(_meta[target+"_censor"])].reset_index(drop=True)
        
        
        F_TCGA = _meta[["histologic_grade", "resection_margin", "lymphovascular_invasion_present", "perineural_invasion_present", "extracapsular_extension", "pathologic_T", "pathologic_N", "pathologic_stage", "localization", "HPV", "metastatic_lymph_node_ratio"]]
        F_GPT = _new_feature[["histologic_grade", "resection_margin", "lymphovascular_invasion_present", "perineural_invasion_present", "extracapsular_extension", "pathologic_T", "pathologic_N", "pathologic_stage", "localization", "HPV", "metastatic_lymph_node_ratio"]]
        tmp = pd.concat((F_TCGA, F_GPT))
        tmp = pd.get_dummies(tmp, columns=["localization"], dtype=int)
        tmp = tmp.drop("localization_0.0", axis=1)
        # tmp = tmp.rename({"localization_0.0":"localization_missing"}, axis=1)
        F_TCGA = tmp.iloc[:len(F_TCGA),:]
        F_GPT = tmp.iloc[len(F_TCGA):,:]

        M = _meta[["age", "gender", "alcohol", "smoking", "pathologic_M", "postoperative_rx_tx"]]
        A = _new_feature[["extranodal_extension", "histotype"]]

        # X = M
        # X = F_TCGA
        # X = F_GPT
        X = pd.concat((M, F_TCGA), axis=1)
        # X = pd.concat((M, F_GPT), axis=1)
        # X = pd.concat((M, F_TCGA, A), axis=1)
        # X = pd.concat((M, F_GPT, A), axis=1)

        X_train = X.loc[_meta['split_'+str(r)] == "train"].reset_index(drop=True)
        embeddings_train = _embeddings.loc[_meta['split_'+str(r)] == "train"].reset_index(drop=True)
        y_train = _meta.loc[_meta['split_'+str(r)] == "train", [target, target+"_censor"]].reset_index(drop=True)
        X_test = X.loc[_meta['split_'+str(r)] == "test"].reset_index(drop=True)
        embeddings_test = _embeddings.loc[_meta['split_'+str(r)] == "test"].reset_index(drop=True)
        y_test = _meta.loc[_meta['split_'+str(r)] == "test", [target, target+"_censor"]].reset_index(drop=True)

        from sklearn.decomposition import PCA
        num_pc = 10
        pca = PCA(n_components=num_pc)
        embeddings_transformed_train = pca.fit_transform(embeddings_train)
        embeddings_transformed_test = pca.transform(embeddings_test)
        print(pca.explained_variance_ratio_.cumsum())

        X_train = pd.concat((X_train, pd.DataFrame(embeddings_transformed_train, columns=["embedding_pc"+str(i) for i in range(num_pc)])), axis=1)
        X_test = pd.concat((X_test, pd.DataFrame(embeddings_transformed_test, columns=["embedding_pc"+str(i) for i in range(num_pc)])), axis=1)
        print(X.shape, X_train.shape, X_test.shape)

        p_values = []
        for i in range(X_train.shape[1]):
            if X_train.iloc[:,[i]].var().iloc[0] == 0:
                continue
            df_train = pd.concat((X_train.iloc[:,[i]],y_train), axis=1)
            cph = CoxPHFitter()
            cph.fit(df_train, duration_col=target, event_col=target+'_censor')

            # Print the summary of the model
            p_values.append([i, cph.summary.p.item()])
        p_values = np.array(p_values)
        idx = p_values[false_discovery_control(p_values[:,1]) < 0.05, 0].astype(int)

        selected_features.append(X_train.columns[idx].tolist())
        # print("significant columns: ", X_train.columns[idx].tolist())
    
        df_train = pd.concat((X_train.iloc[:,idx],y_train), axis=1)
        df_test = pd.concat((X_test.iloc[:,idx],y_test), axis=1)
        cph = CoxPHFitter()
        cph.fit(df_train, duration_col=target, event_col=target+'_censor')

        # Compute the C-index
        c_index = concordance_index(df_test[target], -cph.predict_partial_hazard(df_test), df_test[target+'_censor'])
        # print("C-index: ", c_index)
        cindexs.append(c_index)
    print(target, np.mean(cindexs), np.std(cindexs))
    print(cindexs)
    print(selected_features)





# # Lasso/Elastic - New feature + Raw embedding / Prompt embedding
# l1_ratio=0.5
# for target in ["OS", "DSS", "PFS"]:
#     cindexs = []
#     for r in range(5):
#         _embeddings = embeddings[~pd.isna(meta[target])].reset_index(drop=True)
#         _new_feature = new_feature[~pd.isna(meta[target])].reset_index(drop=True)
#         _meta = meta[~pd.isna(meta[target])].reset_index(drop=True)
#         _embeddings = _embeddings[~pd.isna(_meta[target+"_censor"])].reset_index(drop=True)
#         _new_feature = _new_feature[~pd.isna(_meta[target+"_censor"])].reset_index(drop=True)
#         _meta = _meta[~pd.isna(_meta[target+"_censor"])].reset_index(drop=True)
        
        
#         F_TCGA = _meta[["histologic_grade", "resection_margin", "lymphovascular_invasion_present", "perineural_invasion_present", "extracapsular_extension", "pathologic_T", "pathologic_N", "pathologic_stage", "localization", "HPV", "metastatic_lymph_node_ratio"]]
#         F_GPT = _new_feature[["histologic_grade", "resection_margin", "lymphovascular_invasion_present", "perineural_invasion_present", "extracapsular_extension", "pathologic_T", "pathologic_N", "pathologic_stage", "localization", "HPV", "metastatic_lymph_node_ratio"]]
#         tmp = pd.concat((F_TCGA, F_GPT))
#         tmp = pd.get_dummies(tmp, columns=["localization"], dtype=int)
#         tmp = tmp.drop("localization_0.0", axis=1)
#         # tmp = tmp.rename({"localization_0.0":"localization_missing"}, axis=1)
#         F_TCGA = tmp.iloc[:len(F_TCGA),:]
#         F_GPT = tmp.iloc[len(F_TCGA):,:]

#         M = _meta[["age", "gender", "alcohol", "smoking", "pathologic_M", "postoperative_rx_tx"]]
#         A = _new_feature[["extranodal_extension", "histotype"]]

#         # X = M
#         # X = F_TCGA
#         # X = F_GPT
#         # X = pd.concat((M, F_TCGA), axis=1)
#         X = pd.concat((M, F_GPT), axis=1)
#         # X = pd.concat((M, F_TCGA, A), axis=1)
#         # X = pd.concat((M, F_GPT, A), axis=1)

#         X_train = X.loc[_meta['split_'+str(r)] == "train"].reset_index(drop=True)
#         embeddings_train = _embeddings.loc[_meta['split_'+str(r)] == "train"].reset_index(drop=True)
#         y_train = _meta.loc[_meta['split_'+str(r)] == "train", [target, target+"_censor"]].reset_index(drop=True)
#         X_test = X.loc[_meta['split_'+str(r)] == "test"].reset_index(drop=True)
#         embeddings_test = _embeddings.loc[_meta['split_'+str(r)] == "test"].reset_index(drop=True)
#         y_test = _meta.loc[_meta['split_'+str(r)] == "test", [target, target+"_censor"]].reset_index(drop=True)

#         from sklearn.decomposition import PCA
#         pca = PCA(n_components=10)
#         embeddings_transformed_train = pca.fit_transform(embeddings_train)
#         embeddings_transformed_test = pca.transform(embeddings_test)
#         # print(pca.explained_variance_ratio_.cumsum())

#         X_train = pd.concat((X_train, pd.DataFrame(embeddings_transformed_train)), axis=1)
#         X_test = pd.concat((X_test, pd.DataFrame(embeddings_transformed_test)), axis=1)
#         print(X.shape, X_train.shape, X_test.shape)

#         idx = []
#         for i in range(X_train.shape[1]):
#             if X_train.iloc[:,[i]].var().iloc[0] == 0:
#                 continue
#             idx.append(i)

#         df_train = pd.concat((X_train.iloc[:,idx], y_train), axis=1)
#         df_test = pd.concat((X_test.iloc[:,idx], y_test), axis=1)
        
#         # Assuming X_train, y_train are defined as per your setup
#         from sklearn.model_selection import KFold
#         kf = KFold(n_splits=5)  # Define the number of splits for cross-validation
#         penalizer_values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]  # Example range of penalizer values
#         c_indices = []

#         for penalizer in penalizer_values:
#             temp_c_indices = []
#             for train_index, val_index in kf.split(X_train):
#                 # Splitting the data into training and validation sets
#                 X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
#                 y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

#                 # Training the model
#                 cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
#                 df_train_cv = pd.concat((X_train_cv, y_train_cv), axis=1)
#                 cph.fit(df_train_cv, duration_col=target, event_col=target+'_censor')

#                 # Evaluating the model on the validation set
#                 df_val_cv = pd.concat((X_val_cv, y_val_cv), axis=1)
#                 ci = concordance_index(df_val_cv[target], -cph.predict_partial_hazard(df_val_cv), df_val_cv[target+'_censor'])
#                 temp_c_indices.append(ci)

#             # Averaging the C-index across all folds for the current penalizer value
#             avg_ci = np.mean(temp_c_indices)
#             c_indices.append(avg_ci)

#         # Selecting the penalizer value with the highest average C-index
#         optimal_penalizer = penalizer_values[np.argmax(c_indices)]

#         # Retrain the model with the optimal penalizer on the entire training set
#         print("optimal_penalizer: ", optimal_penalizer)
#         cph_optimal = CoxPHFitter(penalizer=optimal_penalizer, l1_ratio=l1_ratio)
#         df_train = pd.concat((X_train, y_train), axis=1)
#         cph_optimal.fit(df_train, duration_col=target, event_col=target+'_censor')

#         # print(sum(cph.summary.coef>0.001))

#         # Compute the C-index
#         c_index = concordance_index(df_test[target], -cph.predict_partial_hazard(df_test), df_test[target+'_censor'])
#         # print("C-index: ", c_index)
#         cindexs.append(c_index)
#     print(target, np.mean(cindexs), np.std(cindexs))
#     # print(selected_features)




# # Final model
# # CoxPH - New feature + Raw embedding / Prompt embedding
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# for i, target in enumerate(["OS", "DSS", "PFS"]):
#     _embeddings = embeddings[~pd.isna(meta[target])].reset_index(drop=True)
#     _new_feature = new_feature[~pd.isna(meta[target])].reset_index(drop=True)
#     _meta = meta[~pd.isna(meta[target])].reset_index(drop=True)
#     _embeddings = _embeddings[~pd.isna(_meta[target+"_censor"])].reset_index(drop=True)
#     _new_feature = _new_feature[~pd.isna(_meta[target+"_censor"])].reset_index(drop=True)
#     _meta = _meta[~pd.isna(_meta[target+"_censor"])].reset_index(drop=True)
    
    

#     F_TCGA = _meta[["histologic_grade", "resection_margin", "lymphovascular_invasion_present", "perineural_invasion_present", "extracapsular_extension", "pathologic_T", "pathologic_N", "pathologic_stage", "localization", "HPV", "metastatic_lymph_node_ratio"]]
#     F_GPT = _new_feature[["histologic_grade", "resection_margin", "lymphovascular_invasion_present", "perineural_invasion_present", "extracapsular_extension", "pathologic_T", "pathologic_N", "pathologic_stage", "localization", "HPV", "metastatic_lymph_node_ratio"]]
#     tmp = pd.concat((F_TCGA, F_GPT))
#     tmp = pd.get_dummies(tmp, columns=["localization"], dtype=int)
#     tmp = tmp.drop("localization_0.0", axis=1)
#     # tmp = tmp.rename({"localization_0.0":"localization_missing"}, axis=1)
#     F_TCGA = tmp.iloc[:len(F_TCGA),:]
#     F_GPT = tmp.iloc[len(F_TCGA):,:]

#     M = _meta[["age", "gender", "alcohol", "smoking", "pathologic_M", "postoperative_rx_tx"]]
#     A = _new_feature[["extranodal_extension", "histotype"]]

#     # X = M
#     # X = pd.concat((M, F_TCGA), axis=1)
#     X = pd.concat((M, F_GPT), axis=1)
#     # X = pd.concat((M, F_GPT, A), axis=1)

#     y = _meta.loc[:,[target, target+"_censor"]]

#     from sklearn.decomposition import PCA
#     num_pc = 10
#     pca = PCA(n_components=num_pc)
#     embeddings_transformed = pca.fit_transform(_embeddings)
#     print(pca.explained_variance_ratio_.cumsum())

#     X = pd.concat((X, pd.DataFrame(embeddings_transformed, columns=["embedding_pc"+str(i) for i in range(num_pc)])), axis=1)

#     if target == "OS":
#         features = ['age', 'embedding_pc3', 'embedding_pc4', 'embedding_pc8', 'extracapsular_extension', 'metastatic_lymph_node_ratio', 'pathologic_N', 'pathologic_T', 'pathologic_stage', 'perineural_invasion_present', 'postoperative_rx_tx', 'resection_margin']
#     elif target == "DSS":
#         features = ['embedding_pc4', 'embedding_pc8', 'extracapsular_extension', 'metastatic_lymph_node_ratio', 'pathologic_N', 'pathologic_T', 'pathologic_stage', 'perineural_invasion_present', 'postoperative_rx_tx', 'resection_margin']
#     elif target == "PFS":
#         features = ['embedding_pc4', 'extracapsular_extension', 'metastatic_lymph_node_ratio', 'pathologic_N', 'resection_margin']

#     df = pd.concat((X.loc[:,features],y), axis=1)
#     cph = CoxPHFitter()
#     cph.fit(df, duration_col=target, event_col=target+'_censor')

#     # Compute the C-index
#     c_index = concordance_index(df[target], -cph.predict_partial_hazard(df), df[target+'_censor'])

#     cph.plot(ax=axes[i])
#     print(target, c_index)

# plt.tight_layout()
# plt.savefig("fig/coxph.png")
# plt.close()

