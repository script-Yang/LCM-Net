## Gene

- `data_preprocessing/generate_splits.py`  
  Generate dataset splits (e.g., 5-fold cross-validation), following the implementation in  
  [MOTCAT](https://github.com/Innse/MOTCat).

- `data_preprocessing/preprocess_gene_survival.py`  
  Preprocess gene expression data and survival information into tabular format.

- `data_preprocessing/convert_h5_to_parquet.py`  
  Convert raw gene data from `.h5` format into processed parquet files for efficient loading.


## Pathology (WSI Feature Extraction)

We follow the standard [CLAM](https://github.com/mahmoodlab/CLAM)/[UNI](https://github.com/mahmoodlab/UNI) pipeline to extract patch-level visual features from whole-slide
images (WSIs). CLAM is used for WSI preprocessing and patch management, while UNI provides the
pretrained vision encoder.

```bash
git clone https://github.com/mahmoodlab/CLAM
```

Download the pretrained UNI checkpoint and set the environment variable:

```bash
export UNI_CKPT_PATH=/vip_media/sicheng/DataShare/tmi_re/UNI_pth/pytorch_model_v1.bin
```

Extract patch-level WSI features using the pretrained UNI encoder. The extracted features will be
saved to disk and later used as pathology inputs for downstream survival prediction models.

```bash
CUDA_VISIBLE_DEVICES=1 python extract_features_fp.py \
  --data_h5_dir /vip_media/SharedData/SurvivalPred/VisualEbd/LIHC_re/ \
  --data_slide_dir /vip_media/sicheng/DataShare/pathology/LIHC \
  --feat_dir /vip_media/sicheng/DataShare/tmi_re/UNI_results/UNI_LIHC \
  --batch_size 512 \
  --slide_ext .svs \
  --model_name uni_v1 \
  --csv_path /vip_media/sicheng/DataShare/gene_processed_data/LIHC_data.h5
```

**Arguments:**

* `data_slide_dir`: directory containing raw WSIs
* `data_h5_dir`: directory containing preprocessed patch metadata
* `feat_dir`: output directory for extracted WSI features
* `model_name`: pretrained vision encoder (UNI)
* `csv_path`: patient-level metadata used to align WSIs with gene and survival data
