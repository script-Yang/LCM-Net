import pandas as pd
import os
dataset_name = "CESC"
Clinical_df = pd.read_csv(f'/home/data/DataShare/gene_raw_data/survivalcsv.csv')

gene_df = pd.read_csv(f'/home/data/DataShare/gene_raw_data/{dataset_name}.csv')
svs_path = f'/home/data/DataShare/pathology_ans/{dataset_name}_re/ptfiles/pt_files'
file_list = os.listdir(svs_path)

svs = file_list
csv_save_dir = f'/home/data/DataShare/gene_processed_data/{dataset_name}_data.csv'
h5_save_dir = f'/home/data/DataShare/gene_processed_data/{dataset_name}_data.h5'

df = Clinical_df 
df['days_to_event'] = df['death_days_to'].combine_first(df['last_contact_days_to'])
df['vital_status'] = df['vital_status'].replace({'Alive': 0, 'Dead': 1})
df = df[df['vital_status'].isin([0, 1])]
df_subset = df[['bcr_patient_barcode', 'vital_status', 'days_to_event']]
df_subset.rename(columns={
    'bcr_patient_barcode': 'case_id',
    'vital_status': 'censorship',
    'days_to_event': 'survival_days'
}, inplace=True)
df = df_subset

filtered_df = pd.DataFrame()
for f in svs:
    f_name = f[:12]
    if f_name in df['case_id'].values:
        temp_df = df.loc[df['case_id'] == f_name].copy()
        temp_df['slide_id'] = f  
        filtered_df = pd.concat([filtered_df, temp_df], ignore_index=True)

print("**********SURVIVAL DATA SUCCESS**********")
print(filtered_df)
print("**********SURVIVAL DATA SUCCESS**********")

full_data = filtered_df
df = gene_df
columns = df.columns
col_dict = {}
for col in columns[1:]:
    col_name = col[:12]
    col_dict[col_name] = df[col]
df = pd.DataFrame(col_dict)
gene_data = df.T
gene_data.columns = ['gene' + str(i) + 'data' for i in range(1, gene_data.shape[1] + 1)]
gene_data.reset_index(inplace=True)
gene_data.rename(columns={'index': 'case_id'}, inplace=True)
merged_data = pd.merge(full_data, gene_data, on='case_id', how='left')

for col in merged_data.columns:
    if merged_data[col].dtype == 'object' and col != 'case_id' and col != 'slide_id':
        merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')

print("**********MEREGED DATA SUCCESS**********")
df_hdf = merged_data
df_cleaned = merged_data
df_cleaned = df_hdf.dropna(axis=0, thresh=len(df_hdf.columns) - 10)
df_cleaned = df_cleaned.dropna(axis=1, thresh=df_cleaned.shape[0] - 10)
df_cleaned = df_cleaned.reset_index(drop=False)
merged_data = df_cleaned
merged_data = merged_data.dropna()
print(merged_data)

print("**********SURVIVAL DATA SUCCESS**********")
merged_data.to_hdf(h5_save_dir, key='df', mode='w')