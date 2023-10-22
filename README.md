<h1 align="center"> Deconstructing Data Reconstruction: <br>
Multiclass, Weight Decay and General Losses </h1>

<h3 align="center"> 
<a href="https://scholar.google.com/citations?user=YZHL8N0AAAAJ" target="_blank">Gon Buzaglo</a>*, 
<a href="https://nivha.github.io/" target="_blank">Niv Haim</a>*, 
<a href="https://scholar.google.co.il/citations?user=opVT1qkAAAAJ&hl=iw" target="_blank">Gilad Yehudai</a>,
<a href="https://scholar.google.co.il/citations?user=LVk3xE4AAAAJ&hl=en" target="_blank">Gal Vardi</a>,
<a href="https://www.linkedin.com/in/yakir-oz-443aab172?originalSubdomain=il" target="_blank">Yakir Oz</a>,
<a href="https://yaniv.nikankin.com/" target="_blank">Yaniv Nikankin</a>,
<a href="https://www.weizmann.ac.il/math/irani/" target="_blank">Michal Irani</a>
</h3>

<h4 align="center"> NeurIPS 2023 </h4>

## 
Pytorch implementation of the NeurIPS 2023 paper: [Deconstructing Data Reconstruction:
Multiclass, Weight Decay and General Losses](https://arxiv.org/abs/2307.01827).

#### 

## Setup

Create a copy of ```setting.default.py``` with the name ```setting.py```, and make sure to change the paths inside to match your system. 

Create and initialize new conda environment using the supplied ```environment.yml``` file (using python 3.8 and pytorch 11.1.0 and CUDA 11.3) :
```
conda env create -f environment.yaml
conda activate rec
```


## Running the Code

### Notebooks
For quick access, start by running the provided notebooks for analysing the (already provided) 
reconstructions of two (provided) models for multiclass CIFAR10 with CE loss, 
and binary CIFAR10 (vehicles/animals) with MSE loss:

- ```reconstruction_multiclass.ipynb```
- ```reconstruction_regression.ipynb```


### Reproducing the provided trained models and their reconstructions

All training/reconstructions are done by running ```Main.py``` with the right parameters.  
Inside ```command_line_args``` directory we provide command-lines with necessary arguments 
for reproducing the training of the provided models and their provided reconstructions
(those that are analyzed in the notebooks)  


#### Training
For reproducing the training of the provided two trained MLP models (with architecture D-1000-1000-10, and D-1000-1000-1):

 - multiclass model (for reproduction run ```command_line_args/train_cifar10_multiclass_args.txt```)
 - regression model (for reproduction run ```command_line_args/train_cifar10_vehicles_animals_regression_args.txt```)

#### Reconstructions

In ```reconstructions``` directory we provide two reconstructions (results of two runs) per each of the models above.

To find the right hyperparameters for reconstructing samples from the above models 
(or any other models in our paper) we used Weights & Biases sweeps.
In general, it is still an open question how to find the right hyperparameters 
for our losses without trial and error.

These reconstructions can be reproduced by running the following commandlines (the right hyperparameter can be found there):

- multiclass: ```command_line_args/reconstruct_multiclass_run1_args.txt``` and ```command_line_args/reconstruct_multiclass_run2_args.txt```
- regression: ```command_line_args/reconstruct_regression_run1_args.txt``` and ```command_line_args/reconstruct_regression_run2_args.txt```


### Training/Reconstructing New Learning Problems

One should be able to train/reconstruct models for new problems by adding a 
new python file under ```problems``` directory.

Each problem file should contain the logic of how to load the data and 
the parameters necessary to build the model. 


## BibTeX

```bib
@article{buzaglo2023deconstructing,
      title={Deconstructing Data Reconstruction: Multiclass, Weight Decay and General Losses}, 
      author={Gon Buzaglo and Niv Haim and Gilad Yehudai and Gal Vardi and Yakir Oz and Yaniv Nikankin and Michal Irani},
      year={2023},
      eprint={2307.01827},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
