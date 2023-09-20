# Influence of MR Coil Design on Deep-learning-based MR Image Reconstruction

## Abstract
**Purpose:** MR receiver coils used in parallel imaging are designed to satisfy specific geometric requirements that typically seek to minimize correlations between coils. These constraints can lead to coils that are bulky and, sometimes, uncomfortable for the patient. Our work investigates the influence of parallel imaging-imposed coil constraints on reconstruction performance and whether theses constraints  can be relaxed when using deep learning (DL)-based image reconstruction methods.

**Theory and Methods:** Traditional and DL-based MR image reconstruction approaches operate in fundamentally different ways: Traditional methods use only the current image data to find a solution whereas DL methods use a wide range of data/target pairs to learn a generalizable set of reconstruction weights. Early evidence suggests that DL methods are less influenced by coil configurations than traditional methods. Two sets of coil profiles were evaluated: a simulation with 1) idealized coil sensitivities and 2) physically representative coil configurations. In both experiments, degree of coil overlap were varied to assess reconstruction performance of CG-SENSE and a DL model at various acceleration factors. 

**Results:** Both experiments show visually and quantitatively the effects of coil configurations with increasing degrees of overlap (increased aliasing and noise amplification in the reconstructed images) are less pronounced when using a DL method compared to CG-SENSE. 

**Conclusion:** Results suggest that DL image reconstruction methods are less influenced by coil configuration than traditional reconstruction methods. This finding suggests that coil design constraints can be relaxed when using DL reconstruction methods, potentially allowing for more patient-friendly coil designs.

## Code
This repository contains the source code for the following paper: Dubljevic N, Moore S, Lauzon ML, Souza R, Frayne R. Influence of MR coil design on deep-learning-based MR image reconstruction. *Submitted to Magn Reson Med*. 2023.

The code was developed using Python 3.10.9 and Pytorch 1.13.1. The .yml file can be used to recreate the conda environment used for this project. If there are any questions or suggestions for this repository, please let me know at natalia.dubljevic@ucalgary.ca

## Data
The data used in this project is publicly available at as part of the [Calgary-Campinas dataset](https://sites.google.com/view/calgary-campinas-dataset/home). Specifically, the 12-channel raw dataset was used.

