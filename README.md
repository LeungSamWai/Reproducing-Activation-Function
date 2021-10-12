# Reproducing Activation Function for Deep Learning
By [Senwei Liang](https://leungsamwai.github.io/), [Liyao Lyu](http://lylyu.com/), [Chunmei Wang](https://scholar.google.com/citations?user=2mw3BDQAAAAJ&hl=en), [Haizhao Yang](https://haizhaoyang.github.io/)

This repository provides the implementation of paper ''Reproducing Activation Function for Deep Learning'' [[paper]](https://arxiv.org/pdf/2101.04844.pdf).

## Introduction

We propose reproducing activation functions to improve deep learning accuracy for various applications ranging from computer vision to scientific computing. The idea is to employ several basic functions and their learnable linear combination to construct neuron-wise data-driven activation functions for each neuron. Armed with RAFs, neural networks can reproduce traditional approximation tools and, therefore, approximate target functions with a smaller number of parameters than traditional NNs. 

## Requirements

- Python 3.6
- [Pytorch](https://pytorch.org/)

## Implementation
We apply our reproducing activation functions in two kinds of applications, data representation and scientific computing. Please check out the corresponding folders *CoordinateDataRepresentation* and *ScientificComputing*.

## Citation
If you find this paper helps in your research, please kindly cite
```
@article{liang2021reproducing,
  title={Reproducing activation function for deep learning},
  author={Liang, Senwei and Lyu, Liyao and Wang, Chunmei and Yang, Haizhao},
  journal={arXiv preprint arXiv:2101.04844},
  year={2021}
}
```

## Acknowledgement
We thank [Vincent Sitzmann](https://github.com/vsitzmann) for his clean and well-organized code of [SIREN](https://github.com/vsitzmann/siren).