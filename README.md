# NLNet for No-reference Image Quality Assessment

The source codes of the proposed NLNet (Non-Local dependency Network) for No-reference Image Quality Assessment (NR-IQA)

## Usage Demo
### Intra-Database Experiments

### Cross-Database Evaluations

### Experiments Settings
#### Intra-Database Experiments
(1) Split the _reference images_ into **60% training**, **20% validation**, and **20% testing**.<br>
(2) **10 random splits of the reference indices** by **setting random seed from 1 to 10**.<br>
(3) The **median** performance on the testing set is reported.
#### Cross-Database Evaluations
(1) _One database_ is used as the **training set**, and _the other databases_ are the **testing sets**.<br>
(2) The performance of the model in the **last epoch** (100 epochs in this work) is reported.

### Trained Models and Database
LIVE, CSIQ, TID2013, and KADID-10k Databases: Download [here]()<br>
Trained Models: Download [here]()

## Overall Framework
<div>
    <div style="text-align:center">
    <img width=100%device-width src="https://github.com/SuperBruceJia/NLNet-IQA/raw/main/overview.png" alt="NLNet">
</div>

## Paper and Presentations
Note:<br>
(1) **Paper** can be downloaded [here](https://shuyuej.com/files/MMSP/MMSP22_Paper.pdf).<br>
(2) **Slide Presentation** can be downloaded [here](https://shuyuej.com/files/MMSP/MMSP22_Slides.pdf).<br>
(3) **Poster Presentation** can be downloaded [here](https://shuyuej.com/files/MMSP/MMSP22_Poster.pdf).

<div>
    <div style="text-align:center">
    <img width=100%device-width src="https://github.com/SuperBruceJia/NLNet-IQA/raw/main/MMSP22_Poster.png" alt="Poster">
</div>


## Citation
If you find our work useful in your research, please consider citing it in your publications. 
We provide a BibTeX entry below.

```bibtex
@INPROCEEDINGS{Jia2022NLNet,
    author={Jia, Shuvue and Chen, Baoliang and Li, Dingquan and Wang, Shiqi},  
    booktitle={2022 IEEE 24th International Workshop on Multimedia Signal Processing (MMSP)},   
    title={No-reference Image Quality Assessment via Non-local Dependency Modeling},   
    year={2022},
    volume={},
    number={},
    pages={01-06},
    doi={10.1109/MMSP55362.2022.9950035}
}
```
