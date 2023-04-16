
# DataMining-Classification
This repository contains code for my solution of the Classification assignment for the Data Mining course.




## Report

The report containing the context of this assignment, details about my solution and testing results, can be found [here](https://github.com/ViktorHura/DataMining-Classification/blob/main/report/report.pdf). 
## Installation 

You need python version `3.9` or equivalent and making a virtual environment is recommended.

```
pip install -r requirements.txt
```

**Important:** `scikit-learn` version `1.2.2` is defined in the requirements file but 
`Step1_preprocessing.py` requires the use of the `missingpy` package which only supports version `1.1.2` of `scikit-learn`.

The preprocessed data is already available in `/data`, but should you choose to run the preprocessing code, then you must downgrade `scikit-learn`.
## Usage

`Step0_EDA.py`, `Step1_preprocessing.py` and `Step3_predict.py` do not require commandline arguments and can simply be run using the the `python` command.

The training and evaluation script has several commandline options which can be found using the following command
```
python Step2_train_evaluate.py --help
```

Additionally, if you wish to use more parallelised implementations of sklearn algorithms, you can run these scripts using [scikit-learn-intelex](https://github.com/intel/scikit-learn-intelex)

```
python -m sklearnex Step2_train_evaluate.py ...
```
