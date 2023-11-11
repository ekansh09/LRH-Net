
# LRH-Net: A Multi-Level Knowledge Distillation Approach for Low-Resource Heart Network

Pytorch implementation for the LRH-Net: A Multi-Level Knowledge Distillation Approach for Low-Resource Heart Network (FAIR, MICCAI 2022).  


## Proposed Student Architecture

![Optimized Neural Network used](https://github.com/ekansh09/LRH-Net/blob/main/Architectures/LRH-Net.png)


## Proposed Knowledge Distillation methods for ECG classification

Parallel-MLKD            |  Sequential-MLKD
:-------------------------:|:-------------------------:
![](https://github.com/ekansh09/LRH-Net/blob/main/Architectures/p-mlkd.png?raw=true)  |  ![](https://github.com/ekansh09/LRH-Net/blob/main/Architectures/s-mlkd.png?raw=true)


## 🔗 Links
[[Arxiv]](https://arxiv.org/abs/2204.08000)
[[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-18523-6_18)


## Installation

Install [anaconda/miniconda](https://docs.conda.io/en/latest/miniconda.html)  
Required packages

```bash
  $ conda env create --name lrhnet --file env.yml
  $ conda activate lrhnet
```
Install [PyTorch](https://pytorch.org/get-started/locally/)  

## Datasets

Download Datasets from [https://moody-challenge.physionet.org/2020](https://moody-challenge.physionet.org/2020/) and [https://moody-challenge.physionet.org/2021](https://moody-challenge.physionet.org/2021/)

## 🚀 Contributions by LRH-Net
Main contributions of this paper are:
1. A real-time cardiovascular disease detection model which is 106x smaller than a large-scale model and 12x times smaller than the existing low-scale model.
2. A Multi-Level knowledge distillation approach to improve the performance of LRH-Net (student model) and to reduce the number of electrodes and input leads data required for the student model .
3. Performed evaluation on a very diverse, publicly available and combination
of multiple datasets to increase its desirability.


## Results

![](https://github.com/ekansh09/LRH-Net/blob/main/Results/Bar-Plot.jpg?raw=true)

## Futher Reseach can focus on:

Further research can focus on lowering the performance gap in low-lead configurations by optimizing the number of steps required to distill majority of the critical information by varying the levels of MLKD so that the classification performance on hard-to-classify diseases does not get severely affected.

## Citation
If you use the code or results in your research, please use the following BibTeX entry.  
```
@InProceedings{10.1007/978-3-031-18523-6_18,
author="Chauhan, Ekansh
and Guptha, Swathi
and Reddy, Likith
and Raju, Bapi",
title="LRH-Net: A Multi-level Knowledge Distillation Approach for Low-Resource Heart Network",
booktitle="Distributed, Collaborative, and Federated Learning, and Affordable AI and Healthcare for Resource Diverse Global Health",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="190--201",
isbn="978-3-031-18523-6"
}

