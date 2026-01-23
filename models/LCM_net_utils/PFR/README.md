## PFR Sampling for UNI Embeddings

This script applies **PFR-based feature resampling** to UNI-generated `.pth` patch embeddings, selecting a representative subset of features for each slide.
It supports **batch processing**, **resume**, and **atomic saving** to avoid corrupted outputs.

```bash
python train_PFR.py --name BRCA
```

```bash
python train_PFR.py --name COAD
```