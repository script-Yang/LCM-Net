import os
import pandas as pd
h5_dir = "Gene_data"
parquet_dir = "Gene_data_parquet"
os.makedirs(parquet_dir, exist_ok=True)

for filename in os.listdir(h5_dir):
    if filename.endswith(".h5"):
        h5_path = os.path.join(h5_dir, filename)
        parquet_path = os.path.join(parquet_dir, filename.replace(".h5", ".parquet"))
        
        try:
            with pd.HDFStore(h5_path, mode='r') as store:
                keys = store.keys()
                if len(keys) == 0:
                    print(f"No datasets found in {filename}")
                    continue
                key = keys[0]  
                df = pd.read_hdf(h5_path, key=key)
            df.to_parquet(parquet_path, engine="pyarrow")
            print(f"Converted: {filename} -> {os.path.basename(parquet_path)}")
        
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")
