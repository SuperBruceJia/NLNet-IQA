# Non-local Modeling for Image Quality Assessment
## Table of Contents
<ul>
    <li><a href="#Installation">Installation</a></li>
    <li><a href="#Experiments-Settings-and-Quick-Start">Experiments Settings and Quick Start</a></li>
    <li><a href="#Trained-Models-and-Benchmark-Databases">[Download] Trained Models and Benchmark Databases</a></li>
    <li><a href="#Evaluation-Metrics">Evaluation Metrics</a></li>
    <li><a href="#Method-Overview">Method Overview</a></li>
    <li><a href="#Paper-and-Presentations">[Download] Paper and Presentations</a></li>
    <li><a href="#Structure-of-the-Code">Structure of the Code</a></li>
    <li><a href="#Citation">Citation</a></li>
    <li><a href="#Contact">Contact</a></li>
</ul>

## Installation
```python
pip install -r requirements.txt
```

## Experiments Settings and Quick Start
### Intra-Database Experiments
#### Experiments Settings
✔︎ Split the _reference images_ into **60% training**, **20% validation**, and **20% testing**.<br>
✔︎ **10 random splits of the reference indices** by **setting random seed `random.seed(random_seed)` from 1 to 10 `args.exp_id`**.<br>
✔︎ The **median** SRCC and PLCC on the testing set are reported.
#### Quick Start
```python
python main.py --database_path Your_Database_Path --database TID2013 --batch_size 4 --gpu 0
```
Note: <br>
(1) Other hyper-parameters can also be modified via `--parameter XXX`, e.g., `--epochs 200` and `--lr 1e-5`.<br>
(2) Hyper-parameters can be found from the `parser` in the [main.py](https://github.com/SuperBruceJia/NLNet-IQA/blob/main/main.py#L73).

### Cross-Database Evaluations
#### Experiments Settings
✔︎ One database is used as the training set, and the other databases are the testing sets.<br>
✔︎ The performance of the model in the **last epoch** (100 epochs in this work) is reported.
#### Quick Start

### Single Distortion Type Evaluation
```python
python TID2013-Single-Distortion.py
```
Note:<br>
(1) Please change the trained models' path and Database path.<br>
(2) The Distortion Type of the Index can be found from original papers: [TID2013](https://www.sciencedirect.com/science/article/pii/S0923596514001490) and [KADID](http://database.mmsp-kn.de/kadid-10k-database.html). 

### Trained Models and Benchmark Databases
Trained Models: Download [here](https://drive.google.com/drive/folders/1K-24RGXyvSUZfnTThQ0CXUf4BgJA_pn7?usp=sharing)<br>
LIVE, CSIQ, TID2013, and KADID-10k Databases: Download [here](https://drive.google.com/drive/folders/1gfBlByg1bpBXQOFZb6LyCttaX4eAf_Eh?usp=sharing)

### Evaluation Metrics
(1) Pearson Linear Correlation Coefficient (**PLCC**): measures the prediction accuracy<br>
(2) Spearman Rank-order Correlation Coefficient (**SRCC**): measures the prediction monotonicity<br>
A short note of the IQA evaluation metrics can be downloaded [here](https://shuyuej.com/files/MMSP/IQA_Evaluation_Metrics.pdf).

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
(1) **Original Paper** can be downloaded [here](https://shuyuej.com/files/MMSP/MMSP22_Paper.pdf).<br>
(2) **Slides Presentation** can be downloaded [here](https://shuyuej.com/files/MMSP/MMSP22_Slides.pdf).<br>
(3) **Poster Presentation** can be downloaded [here](https://shuyuej.com/files/MMSP/MMSP22_Poster.pdf).

<div>
    <div style="text-align:center">
    <img width=100%device-width src="https://github.com/SuperBruceJia/NLNet-IQA/raw/main/MMSP22_Poster.png" alt="Poster">
</div>

## Structure of the Code
At the root of the project, you will see:
```text
├── main.py
├── model
│   ├── layers.py
│   ├── network.py
│   └── solver.py
├── superpixel
│    └── slic.py
├── lib
│   ├── image_process.py
│   ├── make_index.py
│   └── utils.py
├── data_process
│   ├── get_data.py
│   └── load_data.py
├── benchmark
│   ├── CSIQ_datainfo.m
│   ├── CSIQfullinfo.mat
│   ├── KADID-10K.mat
│   ├── LIVEfullinfo.mat
│   ├── TID2013fullinfo.mat
│   ├── database.py
│   └── datainfo_maker.m
├── save_model
    └── README.md
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

## Contact
If you have any question, please drop me an email at shuyuej@ieee.org.
