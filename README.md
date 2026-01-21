# LCM-Net

> **Note:** This repository contains the **initial** public release of the code.  
> Due to current time constraints, the implementation and documentation may be relatively rough at this stage.  
> If you encounter any issues, please open an issue or contact **syang671@connect.hkust-gz.edu.cn**.


## Data Preparation

### Preparing the Raw Data

Please refer to the [TCGA portal](https://portal.gdc.cancer.gov/) to download the raw data, including raw whole-slide images (WSIs) and gene expression data.

### Embedding the WSIs

For WSIs, please refer to [CLAM](https://github.com/mahmoodlab/CLAM) and [UNI](https://github.com/mahmoodlab/UNI) for details on how to represent these gigapixel whole-slide images.

The pathology data structure should be organized as follows:

```
Pathology
-- BRCA
---- masks
---- patches
---- ptfiles    # vision embeddings
---- stitches
```

### Gene Preprocessing Scripts

The data preprocessing scripts are provided under `data_preprocessing/`:

- `data_preprocessing/generate_splits.py`  
  Generate dataset splits (e.g., 5-fold cross-validation), following the implementation in [MOTCAT](https://github.com/Innse/MOTCat).

- `data_preprocessing/preprocess_gene_survival.py`  
  Preprocess gene expression and survival information into tabular data.

- `data_preprocessing/convert_h5_to_parquet.py`  
  Convert raw gene data from `.h5` format to processed parquet files.

### Data Locations for Reproducibility

Gene data (processed tabular-like data):  
e.g., /vip_media/sicheng/DataShare/tmi_re/Gene_data_parquet/{dataset_name}_data.parquet

Pathology feature files (.pt directory):  
e.g., /vip_media/sicheng/DataShare/tmi_re/UNI_results/UNI_{dataset_name}/pt_files

Data splits (refer to the implementation in [MOTCAT](https://github.com/Innse/MOTCat)):  
e.g., /vip_media/sicheng/DataShare/tmi_re/ours/splits/5foldcv

LLaMA-2-7B (HuggingFace format) weights ([official link](https://huggingface.co/meta-llama/Llama-2-7b-hf)):  
e.g., /vip_media/sicheng/DataShare/Llama-2-7b-hf



## Running Experiments



### Multimodal

```bash
CUDA_VISIBLE_DEVICES=1 python main.py --model_type lcmnet --mode coattn
```
### Genomic
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --model_type snn --mode omic
```
### Pathology
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --model_type transmil --mode path 
```

## Expected Results

**Running example**  
An example on the CESC dataset.
![An example](assets/running_example.png)

## Acknowledgments

The LCM-Net codebase adapts components from
[MOTCAT](https://github.com/Innse/MOTCat),
[CLAM](https://github.com/mahmoodlab/CLAM), [CMTA](https://github.com/FT-ZHOU-ZZZ/CMTA), [UNI](https://github.com/mahmoodlab/UNI) and
[GIO](https://github.com/daeveraert/gradient-information-optimization).
We thank the authors for their excellent work.

