# Introduction

This project is to predict whether a used car purchased by auto dealer have serious issues or not.

These instructions will get you the ideas how the features are extracted and how the hyperparameters are selected for the final model. Also they will help you get a copy of the project up and running on your local machine for development and testing purpose.

## Prerequisites

Following libraries are required before you run the ipython notebook in doc.

```
0. numpy: pip install numpy
1. scikit-learn: pip install scikit-learn
2. pandas: pip install pandas
3. matplotlib: pip install matplotlib
4. seaborn: pip install seaborn
5. lightgbm: pip install lightgbm
```

Following libraries are required before you run the source code in src.

```
0. numpy: pip install numpy
1. scikit-learn: pip install scikit-learn
2. pandas: pip install pandas
3. lightgbm: pip install lightgbm
```

## Exploring and Model Building

If you want to know the process to get the final results, please use Jupyter Notebook to open `/doc/DataExplorationAndModelBuilding.ipynb`

```
In terminal, please type "jupyter notebook", and after lauch jupyter notebook, please open the file in the web browser.
```

## Running the evaluation script

If you want to build the model and evaluate the test dataset, please use following commands. The result would be in `output/`

```
cd src
python eval.py
```

## Authors

* **Haoyang Chen**
