# sigma-phase-prediction

Supervised deep learning prediction of the formation enthalpy of complex phases using a DFT database:
the σ-phase as an example

# Description

This repository is a part of a scientific article submitted in Computational Materials Science - 2021
https://arxiv.org/abs/2011.10883

# Abstract

Machine learning (ML) methods are becoming the state-of-the-art in numerous domains, including material sciences.
In this manuscript, we demonstrate how ML can be used to efficiently predict several properties in solid-state chemistry applications, in particular, to estimate the heat of formation of a given complex crystallographic phase (here, the σ-phase, tP30, D8b).
Based on an independent and unprecedented large first principles dataset containing about 10,000 σ-compounds with *n=14* different elements, 
we used a supervised learning approach to predict all the ~500,000 possible configurations.
From a random set of ~1000 samples, predictions are given within a mean absolute error of 23 meV/at (~2 kJ/mol) on the heat of formation and 0.06 on the tetragonal cell parameters. 
We show that deep neural network regression results in a significant improvement in the accuracy of the predicted output compared to traditional regression techniques. 
We also integrated descriptors having physical nature (atomic radius, number of valence electrons), and we observe that they improve the model precision. 
We conclude from our numerical experiments that the learning database composed of the binary-compositions only, plays a major role in predicting the higher degree system configurations.
Our results open a broad avenue to efficient high-throughput investigations of the combinatorial binary computations for multicomponent complex intermetallic phase prediction.

![TOC](https://user-images.githubusercontent.com/41334324/130812785-19e72aa9-2a98-4689-a571-03354d6eb2f6.png)

# Dataset
the two database have been obtained by DFT calculation under same conditions:
* DB : learning database (~10000 compounds, formed by the combinatorial of given systems)
* DB2 : testing databse (~1000 compounds, formeed by random configuration amon the 14^5 ones

The results of all properties perdictions are given into the file:
* sigma-all.zip

