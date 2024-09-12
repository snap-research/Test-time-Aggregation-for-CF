# TAG-CF: Test-time Aggregation for CF

Source code for the paper **[How Does Message Passing Improve Collaborative Filtering?](https://arxiv.org/abs/2404.08660)**
>by [Mingxuan Ju](https://scholar.google.com/citations?user=qNoO67AAAAAJ&hl=en&oi=ao), [William Shiao](https://scholar.google.com/citations?user=TIq-P5AAAAAJ&hl=en&oi=ao), [Zhichun Guo](https://scholar.google.com/citations?user=BOFfWR0AAAAJ&hl=en&oi=ao), [Fanny Ye](https://scholar.google.com/citations?user=egjr888AAAAJ&hl=en&oi=ao), [Yozen Liu](https://scholar.google.com/citations?user=i3U2JjEAAAAJ&hl=en&oi=ao), [Neil Shah](https://scholar.google.com/citations?user=Qut69OgAAAAJ&hl=en&oi=ao) and [Tong Zhao](https://scholar.google.com/citations?user=05cRc-MAAAAJ&hl=en&oi=ao).

The paper proposes TAG-CF which is a test-time aggregation framework that can be utilized as a plug-and-play module to enhance the performance of matrix factorization models.

## 1. Installation

Please install all dependencies using the command:
```
conda create --name <env> --file requirements.txt
```

## 2. Download data

Please download the Movielens-1M data following the [instruction](./dataset/README.md).

## 3. Run experiments

```
# Train a basic matrix factorization model
make run_pipeline MODEL_YAML_PATH=config/MF.yaml DATA_YAML_PATH=config/bpr_ml.yaml

# Train a LightGCN model.
make run_pipeline MODEL_YAML_PATH=config/LGCN.yaml DATA_YAML_PATH=config/bpr_ml.yaml

# Conduct test-time aggregation on matrix factorization model.
make run_pipeline MODEL_YAML_PATH=config/TAGCF_bpr_ml.yaml  DATA_YAML_PATH=config/bpr_ml.yaml
```

## 4. Reference
```
@article{ju2024does,
  title={How Does Message Passing Improve Collaborative Filtering?},
  author={Ju, Mingxuan and Shiao, William and Guo, Zhichun and Ye, Yanfang and Liu, Yozen and Shah, Neil and Zhao, Tong},
  journal={arXiv preprint arXiv:2404.08660},
  year={2024}
}
```

## 4. Contact
Please contact mju2@nd.edu for any questions.
