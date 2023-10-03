# Effect of MR Coil Geometry on Deep-learning-based MR Image Reconstruction

## Abstract
**Purpose:** To investigate whether parallel imaging-imposed geometric coil constraints can be relaxed when using a deep learning (DL)-based image reconstruction method as opposed to a traditional non-DL method.

**Theory and Methods:** Traditional and DL-based MR image reconstruction approaches operate in fundamentally different ways: Traditional methods solve a system of equations derived from the image data whereas DL methods use data/target pairs to learn a generalizable reconstruction model. Two sets of coil profiles were evaluated: 1) idealized and 2) head coil geometries. A DL model is compared to conjugate gradient SENSE (CG-SENSE) through quantitative metrics and visual assessment as coil overlap is increased. 

**Results:** Experiment 1) As coil overlap increases, there is a significant (*p*<0.001) decrease in all metrics for CG-SENSE, and in most cases for DL. The decrease is most pronounced for CG-SENSE, and the DL model significantly outperforms (*p*}<0.001) CG-SENSE for nearly all degree of coil overlap. Visually, coil overlap increases aliasing. Experiment 2) As coil overlap increases, there is a significant (*p*<0.001) decrease in most cases for the majority of metrics with both methods. The decrease is most pronounced for CG-SENSE, and the DL model significantly outperforms CG-SENSE in every case. Visually, coil overlap increases aliasing and noise amplification in CG-SENSE and introduces no clear visual artifacts for the DL model.

**Conclusion:** The DL image reconstruction method produced higher quality images that were more robust to varying coil overlap configurations than CG-SENSE. This suggests that geometric coil design constraints can be relaxed when using DL reconstruction methods.

## Code
This repository contains the source code for the following paper: Dubljevic N, Moore S, Lauzon ML, Souza R, Frayne R. Effect of MR Coil Geometry on Deep-learning-based MR Image Reconstruction. *Submitted to Magn Reson Med*. Sept 30 2023.

The code was developed using Python 3.10.9 and Pytorch 1.13.1. The .yml file can be used to recreate the conda environment used for this project. If there are any questions or suggestions for this repository, please let me know at natalia.dubljevic@ucalgary.ca

## Data
The data used in this project is publicly available at as part of the [Calgary-Campinas dataset](https://sites.google.com/view/calgary-campinas-dataset/home). Specifically, the 12-channel raw dataset was used.

