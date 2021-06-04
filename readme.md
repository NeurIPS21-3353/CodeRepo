# Learning Equivariant Energy Based Models with Equivariant Stein Variational Gradient Descent

This is the anonymous public code repository for the paper _Learning Equivariant Energy Based Models with Equivariant Stein Variational Gradient Descent_ while it is under review at the 2021 NeurIPS conference. If accepted, the repository will be de-anonymized. 

## Abstract
We focus on the problem of efficient sampling and learning of probability densities by incorporating symmetries in probabilistic models. We first introduce Equivariant Stein Variational Gradient Descent algorithm -- an equivariant sampling method based on Stein's identity for sampling from densities with symmetries. Equivariant SVGD explicitly incorporates symmetry information in a density through equivariant kernels which makes the resultant sampler efficient both in terms of sample complexity and the quality of generated samples. Subsequently, we define equivariant energy based models to model invariant densities that are learned using contrastive divergence. By utilizing our equivariant SVGD for training equivariant EBMs, we propose new ways of improving and scaling up training of energy based models. We apply these equivariant energy models for modelling joint densities in regression and classification tasks for image datasets, many-body particle systems and molecular structure generation.  

## Installation
All code was developed using python version 3.8. To install all required packages run the following command:

`pip install -r requirements.txt`

## Rerunning scripts
The following scripts are available to re-run the experiments as described in the paper. The following describes the mapping from section/experiment to script. 

* Sec. 3 experiment with C_4 Gaussian &#8594; sampling_4Gaussians.py
* Sec. 3 experiment with concentric circles &#8594; sampling_circles.py
* Sec. 3 sample efficiency &#8594; ablation_ll_circles.py
* Sec. 3 robustness &#8594; ablation_initialization_4Gaussians.py
* Sec. 4 C_4 Gaussian JEM &#8594; JEM_4Gaussians.py
* Sec. 5 DW-4 &#8594; EBM_DW4.py & EBM_DW4_sample.py
* Sec. 5 QM9 &#8594; EBM_QM9.py & EBM_QM9_sample.py
* Sec. 5 FashionMNIST &#8594; Coming Soon
* App. C1 Concentric spheres &#8594; sampling_spheres.py
* App. D1 Equivariant JEM &#8594; JEM_4Gaussians.py
* App. D2 JEM concentric circles &#8594; EBM_circles.py
* App. E1 Köhler DW-4 &#8594; EBM_DW4.py & EBM_DW4_sample.py (Set `FIXED = False`)

