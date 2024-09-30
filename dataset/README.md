Please download the ML-1M dataset from the official [grouplens webpage](https://files.grouplens.org/datasets/movielens/ml-1m.zip).

For amazon-book, gowalla, and yelp2018 datasets, we merged the training and testing splits provided in the [LightGCN repository](https://github.com/kuandeng/LightGCN) and transformed them into `.dat` format that shares the same data strcture as ML-1M has.

The transformation script can be found in [this jupyter notebook](dataset_preprocess.ipynb).
