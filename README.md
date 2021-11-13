# NN-Ensemble

Setup Instructions:
---
Install [TF-DM](https://github.com/DependableSystemsLab/TF-DM) and run the `setup.sh` script once.


How to run:
---

```
./perform_fi_exp.py -b $bmark -m $model
```
where `$bmark` is the benchmark name and `$model` is the model name.


Example with running experiments on CIFAR-10, on the VGG16 model, injecting 10% mislabelling errors into the training data.

```
cp ./confFiles/label_err-10.yaml ./confFiles/sample.yaml
./perform_fi_exp.py -b cifar10 -m VGG16
```
Please cite the following paper if you find NN-ensemble useful
---
https://blogs.ubc.ca/dependablesystemslab/2021/10/22/understanding-the-resilience-of-neural-network-ensembles-against-faulty-training-data/
