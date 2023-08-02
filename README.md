# Simulated Federated Learning on IID and Non-IID data

Federated Learning (FL) is a great learning paradigm to jointly train machine learning models among various clients while data privacy is maintained.
This project serves to demonstrate the potential of FL on datasets of your choice.
Therefore it provides a variety of FL strategies, types of data imbalance, and allows to apply custom models and data.

Despite its potential, FL is known to suffer from what is called non-iid data. Here, the performance of the jointly trained model decreases if the data held by each client differs in terms of labels, quantity, or features.
In turn, this project provides a toolset to simulate different types of data imbalance within the federation in order to measure its impact on FL performance.

## Installation

The installation works best in a brand conda environment. 
Once created, install the required packages using ___pip install___ or ___conda install___(preferred), using
```
pip install -r requirements.txt
```
or 
```
conda install --file requirements.txt
```

## Run

Use the following command to run FL from your terminal
```
python run.py --strategy="fedavg" --dataset="covtype"
```
Alternatively, you can run the project interactively using the provided ___Jupyter Notebook___.

## Choose Federated Learning Strategies

The following FL strategies are contained and can be applied using the given name:
- ___FedAvg___: McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017, April). Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics (pp. 1273-1282). PMLR.
- ___FedProx___: Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. Proceedings of Machine learning and systems, 2, 429-450.
- ___FedAtt___: Ji, S., Pan, S., Long, G., Li, X., Jiang, J., & Huang, Z. (2019, July). Learning private neural language modeling with attentive aggregation. In 2019 International joint conference on neural networks (IJCNN) (pp. 1-8). IEEE.
- ___FedED___: Sui, D., Chen, Y., Zhao, J., Jia, Y., Xie, Y., & Sun, W. (2020, November). Feded: Federated learning via ensemble distillation for medical relation extraction. In Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP) (pp. 2118-2128).
- ___FedMD___: Li, D., & Wang, J. (2019). Fedmd: Heterogenous federated learning via model distillation. arXiv preprint arXiv:1910.03581.
- ___FedDF___: Lin, T., Kong, L., Stich, S. U., & Jaggi, M. (2020). Ensemble distillation for robust model fusion in federated learning. Advances in Neural Information Processing Systems, 33, 2351-2363.

## Apply Data Imbalances

The project allows to apply two kinds of data imbalance subsequentially:
### Label skew
- ___None___ or ___homogeneous___: no skew is applied. Data distribution is iid.
- ___label_distribution___: clients are forced to differ in terms of their label distribution. The parameter ___label_alpha___ controls for the severeness, where small values indicate high label imbalance, i.e., clients differ strongly w.r.t. their labels held.
- ___label_quantity___: clients are limited to the mount of different labels defined by ___label_n___. Hence, not each client might know about all labels.
- ___quantity___: When this is selected, the parameter ___label_alpha___ controls for the the difference in number of samples held by each client. The labels are distributed iid, but some clients possess more samples than others.
- ___clustering___: relies on a clustering-based approach of client distribution. The parameter ___purity___ defines the extent of imbalance, where high ___purity___ indicates high imbalance.
### Attribute skew
- ___noise___: Randomly inserts noise into the features of individual clients by replacing them, e.g., with zeros. The parameter ___attribute_alpha___ controls for the severeness, where small values indicate high imbalance.
- ___availablity___: zeros-out entire features (randomly selected) per client. The parameter ___attribute_alpha___ controls for the severeness, where small values indicate a large number of columns being removed.


## Add Models

Custom models can be added in the ___models.py___ file.
Therefore, add another class according to the scheme of the exsting classes.
You need to define a ___forward___-pass function and define the all layer in the ___init___.

Finally, register your model in the ___get_model_by_name___ function at the bottom of the file.

## Add datasets

Custom datasets can be added in the ___load_data.py___ file.
Therefore, add the data-file to a new subfolder of ___/data___ and add your dataloader in the ___load_raw_data___ function.

There, you have to set the number of features and classes, define a model if ___model_name___ is set to ___auto___ and read your data accordingly.
Finally, ___X_raw___ has to contain a list of lists, where each list is the feature representation of a single data point.
___y_raw___ should be a list of float or int values, either containing the index of the respective class or the regression target.
