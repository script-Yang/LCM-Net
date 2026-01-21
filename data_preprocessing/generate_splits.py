import pandas as pd
import numpy as np

import os
dataset_name = 'CESC'
hdf_file = f'/home/data/DataShare/gene_processed_data/{dataset_name}_data.h5'
merged_data = pd.read_hdf(hdf_file, 'df')

np.random.seed(42)
shuffled_data = merged_data.sample(frac=1).reset_index(drop=True)


n_splits = 5
fold_size = len(shuffled_data) // n_splits
splits_indices = [list(range(i * fold_size, (i + 1) * fold_size)) for i in range(n_splits)]

for i in range(n_splits):
    val_indices = splits_indices[i]
    train_indices = [idx for idx in range(len(shuffled_data)) if idx not in val_indices]
    val_data = shuffled_data.iloc[val_indices]['case_id'].tolist()
    train_data = shuffled_data.iloc[train_indices]['case_id'].tolist()
    max_length = max(len(train_data), len(val_data))
    train_data.extend([''] * (max_length - len(train_data))) 
    val_data.extend([''] * (max_length - len(val_data)))      
    result_df = pd.DataFrame({
        'train': train_data,
        'val': val_data
    }, index=range(max_length))

    result_df.reset_index(drop=True, inplace=True)
    os.makedirs(f'./splits/5foldcv/{dataset_name}',exist_ok=True)
    result_df.to_csv(f'./splits/5foldcv/{dataset_name}/splits_{i}.csv', header=True)


print(f"***{dataset_name}*** 5-fold cross-validation splits have been saved to splits_0.csv to splits_4.csv")