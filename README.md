# Adult-Census-Income-Classification
In this project, I work with a classical machine learning dataset - the [Adult Census](https://archive.ics.uci.edu/ml/datasets/Adult), to develop a predictive model for whether census individuals have an income that exceeds $50k. I use this problem to explore automated data pipelines and the development of custom sklearn data transformers. 

Using careful feature pre-processing and model tuning, I achieved superior performance - **87.4% accuracy** - to other publicly available models I found. By comparison, the model in the [original paper](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf) that introduced this dataset has accuracy of 85.9%. [Google TensorFlow’s wide-deep models](https://github.com/tensorflow/models/tree/master/official/wide_deep) achieve just over 83% accuracy. 

Included in this repo are:
1. [A Jupyter notebook with exploratory data analysis](https://github.com/tal-yifat/Adult-Census-Income-Classification/blob/master/Exploratory%20Data%20analysis.ipynb).
2. [A Jupyter notebook with the data transformation pipeline and the development of the model](https://github.com/tal-yifat/Adult-Census-Income-Classification/blob/master/Data%20Pipeline%20and%20Models.ipynb).
3. [A Python module with custom data transformers I developed](https://github.com/tal-yifat/Adult-Census-Income-Classification/blob/master/custom_transformers.py).

The most interesting aspect of this project is two custom sklearn data transformers I developed to systematically merge nominal and ordinal categories that have similar statistical relationships with the target feature. The Adult Census dataset originally had nine  categorical features with multiple categories, resulting in a cluttered dataset with many features of very little importance. In the selected model, the data transformers reduced the number of binary one-hot-encoded categories from 100 to 38, and the number of ordinal categories from 16 to 8 (these numbers can be tuned using hyperparameters). The transformers were developed so they can seamlessly integrate with sklearn's automated data pipelines.

The merging algorithm performs a proportion similarity Z test for each pair of mergeable categories and merges the pair with the highest P-value. The P-value here represents the probability, given that two categories have the same distribution of target feature positives, that Z would be greater than or equal to the observed results. In other words, a high P-value indicates that two categories likely have a similar statistical relationship with the target variable. Merged categories need to meet user-determined category size and minimum P-value thresholds. After merging the most similar pair of categories, the algorithm repeats this process until no more pairs meet the merging criteria. 

I am using this project to experiment with new techniques and will occasionally update this repo when I have interesting results.
