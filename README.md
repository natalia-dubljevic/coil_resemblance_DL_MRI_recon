# Effect of MR Head Coil Geometry on Deep-learning-based MR Image Reconstruction

## Abstract
**Purpose:** To investigate whether parallel imaging-imposed geometric coil constraints can be relaxed when using a deep learning (DL)-based image reconstruction method as opposed to a traditional non-DL method.

**Theory and Methods:** Traditional and DL-based MR image reconstruction approaches operate in fundamentally different ways: Traditional methods solve a system of equations derived from the image data whereas DL methods use data/target pairs to learn a generalizable reconstruction model. Two sets of coil profiles were evaluated: 1) 8-channel and 2) 32-channel geometries. A DL model was compared to conjugate gradient SENSE (CG-SENSE) and L1-wavelet compressed sensing (CS) through quantitative metrics and visual assessment as coil overlap was increased.

**Results:** Results were generally consistent between experiments. As coil overlap increased, there was a significant (*p* < 0.001) decrease in performance in most cases for all methods. The decrease was most pronounced for CG-SENSE, and the DL models significantly outperformed (*p* < 0.001) their non-DL counterparts in all scenarios. CS showed improved robustness to coil overlap and SNR vs. CG-SENSE, but had quantitatively and visually poorer reconstructions characterized by blurriness as compared to DL. DL showed virtually no change in performance across SNR and very small changes across coil overlap.

**Conclusion:** The DL image reconstruction method produced images that were robust to coil overlap and of higher quality than CG-SENSE and CS. This suggests that geometric coil design constraints can be relaxed when using DL reconstruction methods.

## Code
This repository contains the source code for the following paper: Dubljevic N, Moore S, Lauzon ML, Souza R, Frayne R. Effect of MR Head Coil Geometry on Deep-learning-based MR Image Reconstruction. *Submitted to Magn Reson Med*. Sept 30 2023.

The code was developed using Python 3.10.9 and Pytorch 1.13.1. The .yml file can be used to recreate the conda environment used for this project. If there are any questions or suggestions for this repository, please let me know at natalia.dubljevic@ucalgary.ca

## Data
The data used in this project is publicly available at as part of the [Calgary-Campinas dataset](https://sites.google.com/view/calgary-campinas-dataset/home). Specifically, the 12-channel raw dataset was used.

