# Age Estimation based on Multi-task Neural Process

Pytorch implementation of [Neural Processes](https://arxiv.org/abs/1807.01622). This repo follows the
best practices defined in [Empirical Evaluation of Neural Process Objectives](http://bayesiandeeplearning.org/2018/papers/92.pdf).

## Computational Graph

some notations: 

​	trainingSet1: for 1st training on whole training set

​	person<sub>1...N</sub> : components of trainSet1 (separated from trainSet1 by each person)

​	trainingSet2: for MergeNet training (completely different with trainingSet1)

### Training Round 1

trainingSet1 ====>  NP model  ====> pretrained model1(PM1)

### Training Round 2

Person1 ====>  PM1  ====> pretrained model2<sub>1</sub> (PM2<sub>1</sub>)

Person2 ====>  PM1  ====> PM2<sub>2</sub>

……

……
PersonN ====>  PM1  ====> PM2<sub>N</sub>

### Training Round 3

trainingSet2  ====>  PM2<sub>1</sub>   ====>   |&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  

trainingSet2  ====>  PM2<sub>2</sub>   ====>   |&nbsp;&nbsp;&nbsp;&nbsp;Merge&nbsp;&nbsp;&nbsp;&nbsp;|

……&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Net&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  ====>   estimating age

……&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|

trainingSet2  ====>  PM2<sub>N</sub>   ====>  |&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|


## Current Demo

Our current progress can be seen in multi-task-example-ageEstimation.ipynb.

## TODO

The model is just a toy model now.

we haven't finetune it and maybe we should add more hidden layers to improve the performance

