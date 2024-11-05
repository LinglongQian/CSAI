# CSAI: Knowledge Enhanced Conditional Imputation for Healthcare Time-series

Official implementation of "Knowledge Enhanced Conditional Imputation for Healthcare Time-series"

## Overview

CSAI (Conditional Self-Attention Imputation) is a novel recurrent neural network architecture designed to handle complex missing data patterns in multivariate time series from electronic health records (EHRs). Key features:

- Domain-informed temporal decay mechanism adapted to clinical data recording patterns
- Attention-based hidden state initialization for capturing long and short-range dependencies
- Non-uniform masking strategy that reflects real-world missingness patterns

## Model Architecture

![CSAI Architecture](figures/CSAI.png)

## Installation

```bash
# Clone the repository
git clone https://github.com/LinglongQian/CSAI.git
cd CSAI

# Create conda environment
conda env create -f csai.yml

# Activate environment
conda activate csai
```

## Dataset Preparation

The implementation supports three healthcare benchmark datasets:

1. **PhysioNet Challenge 2012**
   - 4,000 ICU stays with 35 variables
   - [Download link](https://physionet.org/content/challenge-2012/1.0.0/)

2. **MIMIC-III**
   - 59 variables benchmark
   - Requires [credentialed access](https://physionet.org/content/mimiciii/1.4/)

3. **eICU**
   - 20 variables benchmark
   - Available after [registration](https://physionet.org/content/eicu-crd/2.0/)

Place the downloaded data in the `data/` directory following this structure:
```
data/
├── physionet/
├── mimic_59f/
└── eicu/
```

## Usage

### Data Preprocessing
```bash
python process_physionet.py --data_dir ./data/physionet --output_dir ./data/physionet [--n_splits 5] [--seed 3407]
```

### Training example
```bash
python main.py \
    --dataset physionet \
    --model_name CSAI \
    --task I \  # I for imputation, C for classification
    --gpu_id 0 \
    --epoch 300 \
    --lr 0.0005 \
    --batchsize 64
```

## Citation

```bibtex
@article{qian2023knowledge,
  title={Knowledge Enhanced Conditional Imputation for Healthcare Time-series},
  author={Qian, Linglong and Raj, Joseph Arul and Ellis, Hugh Logan and Zhang, Ao and Zhang, Yuezhou and Wang, Tao and Dobson, Richard JB and Ibrahim, Zina},
  journal={arXiv preprint arXiv:2312.16713},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This research was supported by [funding details from paper]. The codebase builds upon several excellent repositories:
- [BRITS](https://github.com/caow13/BRITS)
- [PyPOTS](https://github.com/WenjieDu/PyPOTS)
