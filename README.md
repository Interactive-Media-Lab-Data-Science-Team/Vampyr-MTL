# Vampyr-MTL

![Python](https://img.shields.io/badge/Python-^3.6-blue.svg?logo=python&longCache=true&logoColor=white&colorB=5e81ac&style=flat-square&colorA=4c566a)
![Pandas](https://img.shields.io/badge/Pandas-v1.0.4-blue.svg?longCache=true&logo=python&longCache=true&style=flat-square&logoColor=white&colorB=5e81ac&colorA=4c566a)
![Plotly](https://img.shields.io/badge/Plotly-v4.8.1-blue.svg?longCache=true&logo=python&longCache=true&style=flat-square&logoColor=white&colorB=5e81ac&colorA=4c566a)
![Numpy](https://img.shields.io/badge/Numpy-v1.18.0-blue.svg?longCache=true&logo=python&longCache=true&style=flat-square&logoColor=white&colorB=5e81ac&colorA=4c566a)
![Scikit-learn](https://img.shields.io/badge/sklearn-v0.23.0-blue.svg?longCache=true&logo=python&longCache=true&style=flat-square&logoColor=white&colorB=5e81ac&colorA=4c566a)
![GitHub Last Commit](https://img.shields.io/github/last-commit/google/skia.svg?style=flat-square&colorA=4c566a&colorB=a3be8c)
![GitHub Issues](https://img.shields.io/github/issues/Interactive-Media-Lab-Data-Science-Team/Vampyr-MTL)
![GitHub Stars](https://img.shields.io/github/stars/Interactive-Media-Lab-Data-Science-Team/Vampyr-MTL)
![GitHub Forks](https://img.shields.io/github/forks/Interactive-Media-Lab-Data-Science-Team/Vampyr-MTL)
![Github License](https://img.shields.io/github/license/Interactive-Media-Lab-Data-Science-Team/Vampyr-MTL)
---
![Vampire Squid](/package_info/logo_large.png "Vampyr Inspiration")

**Vampyr-MTL** is a machine learning python package inspired by [MALSAR](https://github.com/jiayuzhou/MALSAR) multi-task learning Matlab algorithm, combined with up-to-date multi-task learning researches and algorithm for public research purposes.
## [Demo](https://colab.research.google.com/drive/1SVMAEnu1Jk4ldvUqI5iuA7D1zlxXrLrr?usp=sharing)
Demo for runing Clustered Multitask Learning algorithm with risk factor analysis, pls copy to your playground and do not ask for change authorizations

## FunctionalityMarkdown Preview EnhancedMarkdown Preview Enhanced
* Algorithms:
  - Multitask Binary Logistic Regression
    + Hinge Loss 
    + L21 normalization
  - Multitask Linear Regression
    + Mean Square Error
    + L21 normalization
  - Cluster Multitask Least Square Regression
    + L21 Normalization
* Util Functions:
  - MTL_data_split
    + Split data set inside each task with predefined proportions, build on sklearn train_test_split
  - MTL_data_extract
    + Extract data from pandas.DataFrame to desired data matrix format, with desired target and task
  - Cross Validation with k Folds:
    + Cross validation with predefined k folds and scoring methods
    
more see [*Documentation*](https://vampyr-mtl.readthedocs.io/en/latest/)

## Related Reseaches
[Accelerated Gredient Method](https://arxiv.org/pdf/1310.3787.pdf)

[Clustered Multi-Task Learning: a Convex Formulation](https://papers.nips.cc/paper/3499-clustered-multi-task-learning-a-convex-formulation.pdf)

[Regularized Multi-task Learning](https://dl.acm.org/doi/pdf/10.1145/1014052.1014067)

## Installation (test version)
``pip install -i https://test.pypi.org/simple/ Vampyr-MTL-Max-JJ==0.0.5``

## Dependency
Auto generated by [pigar](https://github.com/damnever/pigar)
- scikit_learn == 0.22.1

- setuptools == 45.2.0

- tqdm == 4.46.1

- plotly == 4.8.1

- numpy == 1.18.1

- pandas == 1.0.4

- pytest == 5.3.5

- scipy == 1.4.1

## Package Update

* Manual Deployment:

  - [test-pypi manual](https://packaging.python.org/tutorials/packaging-projects/)

  - ``python3 setup.py sdist bdist_wheel``

  - ``python3 -m twine upload --repository testpypi dist/*``

  or rewrite .pypirc file with credencials and 

  - ``python3 twine upload -r pypi dist/*``

  - ``python3 setup.py dist bdist_wheel``

* Automation(Linux):
  - deploy: ``./build_deploy.sh``
  - test: ``./build_deploy.sh --test``

## Development

*Windows*
```$ git clone https://github.com/Interactive-Media-Lab-Data-Science-Team/Vampyr-MTL.git

$ cd Vampyr_MTL

$ python3 -m venv myenv

$ myenv/Scripts/activate

$ pip3 install -r requirements.txt
```
