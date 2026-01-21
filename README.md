# LCM-Net

> **Note:** This repository contains the initial public release of the code.  
> Due to current time constraints, the implementation and documentation may be relatively rough at this stage.  
> If you encounter any issues, please open an issue or contact **syang671@connect.hkust-gz.edu.cn**.



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
[CLAM](https://github.com/mahmoodlab/CLAM), and
[GIO](https://github.com/daeveraert/gradient-information-optimization).
We thank the authors for their excellent work.

