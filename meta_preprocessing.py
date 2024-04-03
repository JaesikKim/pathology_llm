import pandas as pd
import numpy as np

# comparison between meta and new feature
meta = pd.read_csv("/project/kimlab_hnsc/data/text_files/deduplicated_hnsc_with_ratio.csv")
print(meta.shape)
# meta['pathologic_TX'] = (meta['pathologic_T'] == 'TX').astype(int)
# meta['pathologic_NX'] = (meta['pathologic_N'] == 'NX').astype(int)

meta['pathology_report_file_name'] = [i[:-4].upper() for i in meta['pathology_report_file_name']]
meta['gender.demographic'] = meta['gender.demographic'].replace({np.nan:0, 'female':1, 'male':2})
meta['alcohol_history.exposures'] = meta['alcohol_history.exposures'].replace({np.nan:0, 'Not Reported':0, 'No':1, 'Yes':2})
meta['pathologic_M'] = meta['pathologic_M'].replace({np.nan:0, 'MX':1, 'M0':1, 'M1':2})
meta['postoperative_rx_tx'] = meta['postoperative_rx_tx'].replace({np.nan:0, 'NO':1, 'YES':2})

meta['histologic_grade'] = meta['histologic_grade'].replace({np.nan:0, 'GX':0, 'G1':1,'G2':2,'G3':3,'G4':3})
meta['Resection_margin_status'] = meta['Resection_margin_status'].replace({np.nan:0, "Negative":1, 'Close':2, 'Positive':3})
meta['LVI'] = meta['LVI'].replace({np.nan:0, 'NO':1, 'YES':2})
meta['PNI'] = meta['PNI'].replace({np.nan:0, 'NO':1, 'YES':2})
meta['Extracapsular_extension'] = meta['Extracapsular_extension'].replace({np.nan:0, 'No Extranodal Extension':1, 'Microscopic Extension':2, 'Gross Extension':3})
meta['pathologic_T'] = meta['pathologic_T'].replace({np.nan:0, 'TX':0, 'T0':1, 'T1':2, 'T2':3, 'T3':4, 'T4a':5, 'T4b':6, 'T4':5})
meta['pathologic_N'] = meta['pathologic_N'].replace({np.nan:0, 'NX':0, 'N0':1, 'N1':2, 'N2':3, 'N2a':3, 'N2b':3, 'N2c':3, 'N3':4})
meta['pathologic_stage'] = meta['pathologic_stage'].replace({np.nan:0, '[Discrepancy]':0, 'Stage I':1, 'Stage II':2, 'Stage III':3, 'Stage IVA':4, 'Stage IVB':4, 'Stage IVC':4})
# meta['tobacco_smoking_history'] = meta['tobacco_smoking_history'].replace({np.nan:0, 1:1, 2:2, 3:2, 4:2, 5:2})
meta['tobacco_smoking_history'] = meta['tobacco_smoking_history'].replace({np.nan:0, 1:1, 2:3, 3:2, 4:2, 5:2})
meta['subsite'] = meta['subsite'].replace({np.nan:0, 'OSCC':1, 'OPSCC':2, 'Larynx':3, 'Hypopharynx':4})
meta['HPV_status_x'] = meta['HPV_status_x'].replace({np.nan:0, 'HNSC_HPV-':1, 'HNSC_HPV+':2})
meta['metastatic_lymph_node_ratio'] = meta['number_of_lymphnodes_positive_by_he']/meta['lymph_node_examined_count']

meta['Disease.specific.Survival.status'] = meta['Disease.specific.Survival.status'].replace({'0:ALIVE OR DEAD TUMOR FREE':0, '1:DEAD WITH TUMOR':1})
meta['Overall.Survival.Status'] = meta['Overall.Survival.Status'].replace({'0:LIVING':0, '1:DECEASED':1})
meta['Progression.Free.Status'] = meta['Progression.Free.Status'].replace({'0:CENSORED':0, '1:PROGRESSION':1})

meta = meta[["pathology_report_file_name", "Age", "gender.demographic", "alcohol_history.exposures", "tobacco_smoking_history", "pathologic_M", "postoperative_rx_tx",
             "histologic_grade", "Resection_margin_status", "LVI", "PNI", "Extracapsular_extension", "pathologic_T", "pathologic_N", "pathologic_stage", "subsite", "HPV_status_x", "metastatic_lymph_node_ratio",
             "Months.of.disease.specific.survival", "Disease.specific.Survival.status", "Overall.Survival..Months.", "Overall.Survival.Status", "Progress.Free.Survival..Months.", "Progression.Free.Status"]]

meta.columns = ["sample_id", "age", "gender", "alcohol", "smoking", "pathologic_M", "postoperative_rx_tx",
                "histologic_grade", "resection_margin", "lymphovascular_invasion_present", "perineural_invasion_present", "extracapsular_extension", 
                "pathologic_T", "pathologic_N", "pathologic_stage", "localization", "HPV", "metastatic_lymph_node_ratio",
                "DSS","DSS_censor","OS","OS_censor","PFS","PFS_censor"]

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
for i, (train_index, test_index) in enumerate(skf.split(meta, meta.OS_censor)):
    meta['split_'+str(i)] = ''
    for j in train_index:
        meta.loc[j, 'split_'+str(i)] = 'train'
    for j in test_index:
        meta.loc[j, 'split_'+str(i)] = 'test'

meta.to_csv("/project/kimlab_hnsc/data/text_files/processed_deduplicated_hnsc_with_ratio.csv", index=False, header=True)
