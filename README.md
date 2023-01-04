# Non-local Modeling for Image Quality Assessment

The source codes of the proposed NLNet (Non-Local dependency Network) for No-reference Image Quality Assessment (NR-IQA)

## Installation

## Experiments Settings and Usage Demo
### Intra-Database Experiments
#### Experiments Settings
(1) Split the _reference images_ into **60% training**, **20% validation**, and **20% testing**.<br>
(2) **10 random splits of the reference indices** by **setting random seed `random.seed(random_seed)` from 1 to 10**.<br>
(3) The **median** SRCC/PLCC performances on the testing set are reported.
#### Quick Start

### Cross-Database Evaluations
#### Experiments Settings
(1) _One database_ is used as the **training set**, and _the other databases_ are the **testing sets**.<br>
(2) The performance of the model in the **last epoch** (100 epochs in this work) is reported.
#### Quick Start

### Trained Models and Database
LIVE, CSIQ, TID2013, and KADID-10k Databases: Download [here](https://drive.google.com/drive/folders/1gfBlByg1bpBXQOFZb6LyCttaX4eAf_Eh?usp=sharing)<br>
Trained Models: Download [here]()

## Method Overview
<div>
    <div style="text-align:center">
    <img width=100%device-width src="https://github.com/SuperBruceJia/NLNet-IQA/raw/main/overview.png" alt="NLNet">
</div>

(i) **Image Preprocessing**: The input image is pre-processed.<br>
(ii) **Graph Neural Network – Non-Local Modeling Method**: A two-stage GNN approach is presented for the non-local feature extraction and long-range dependency construction among different regions. The first stage aggregates local features inside superpixels. The following stage learns the non-local features and long-range dependencies among the graph nodes. It then integrates short- and long-range information based on an attention mechanism. The means and standard deviations of the non-local features are obtained from the graph feature signals.<br>
(iii) **Pre-trained VGGNet-16 – Local Modeling Method**: Local feature means and standard deviations are derived from the pre-trained VGGNet-16 considering the hierarchical degradation process of the HVS.<br>
(iv) **Feature Mean & Std Fusion and Quality Prediction**: The means and standard deviations of the local and non-local features are fused to deliver a robust and comprehensive representation for quality assessment. Besides, the distortion type identification loss Lt , quality prediction loss Lq , and quality ranking loss Lr are utilized for training the NLNet. During inference, the final quality of the image is the averaged quality of all the non-overlapping patches.

## Paper and Presentations
Note:<br>
(1) **Paper** can be downloaded [here](https://shuyuej.com/files/MMSP/MMSP22_Paper.pdf).<br>
(2) **Slide Presentation** can be downloaded [here](https://shuyuej.com/files/MMSP/MMSP22_Slides.pdf).<br>
(3) **Poster Presentation** can be downloaded [here](https://shuyuej.com/files/MMSP/MMSP22_Poster.pdf).

<div>
    <div style="text-align:center">
    <img width=100%device-width src="https://github.com/SuperBruceJia/NLNet-IQA/raw/main/MMSP22_Poster.png" alt="Poster">
</div>

## Structure of the code
At the root of the project, you will see:

```text
├── 
```

## Citation
If you find our work useful in your research, please consider citing it in your publications. 
We provide a BibTeX entry below.

```bibtex
@INPROCEEDINGS{Jia2022NLNet,
    author    = {Jia, Shuvue and Chen, Baoliang and Li, Dingquan and Wang, Shiqi},  
    booktitle = {2022 IEEE 24th International Workshop on Multimedia Signal Processing (MMSP)},   
    title     = {No-reference Image Quality Assessment via Non-local Dependency Modeling},   
    year      = {Sept. 2022},
    volume    = {},
    number    = {},
    pages     = {01-06},
    doi       = {10.1109/MMSP55362.2022.9950035}
}
```
