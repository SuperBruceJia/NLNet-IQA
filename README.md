# Non-local Modeling for Image Quality Assessment
<div>
    <div style="text-align:center">
    <img width=100%device-width src="https://github.com/SuperBruceJia/NLNet-IQA/raw/main/overview.png" alt="NLNet">
</div>

## Table of Contents
<ul>
    <li><a href="#Installation">Installation</a></li>
    <li><a href="#Experiments-Settings-and-Quick-Start">Experiments Settings and Quick Start</a></li>
    <li><a href="#Superpixel-Segmentation-Demo">Superpixel Segmentation Demo</a></li>
    <li><a href="#Trained-Models-and-Benchmark-Databases">Trained Models and Benchmark Databases</a></li>
    <li><a href="#Global-Distortions-and-Local-Distortions">Global Distortions and Local Distortions</a></li>
    <li><a href="#Evaluation-Metrics">Evaluation Metrics</a></li>
    <li><a href="#Paper-and-Presentations">Paper and Presentations</a></li>
    <li><a href="#Structure-of-the-Code">Structure of the Code</a></li>
    <li><a href="#Citation">Citation</a></li>
    <li><a href="#Contact">Contact</a></li>
    <li><a href="#Acknowledgement">Acknowledgement</a></li>
</ul>

## Installation
Framework: PyTorch, OpenCV, PIL, scikit-image, scikit-learn, Numba JIT, Matplotlib, etc.<br>
**Note**: The overall framework is based on **PyTorch**. Here, I didn't provide a specific `pip install -r requirements.txt` because there are so many dependencies. I would like to suggest you install the corresponding packages when they are required to run the code.

## Experiments Settings and Quick Start
### Intra-Database Experiments
Experiments Settings:<br>
✔︎ Split the reference images into 60% training, 20% validation, and 20% testing.<br>
✔︎ 10 random splits of the reference indices by setting random seed `random.seed(random_seed)` from 1 to 10 `args.exp_id`.<br>
✔︎ The median SRCC and PLCC on the testing set are reported.<br>

Quick Start:<br>
```python
python main.py --database_path '/home/jsy/BIQA/' --database TID2013 --batch_size 4 --num_workers 8 --gpu 0
```
(1) Other hyper-parameters can also be modified via `--parameter XXX`, e.g., `--epochs 200` and `--lr 1e-5`.<br>
(2) Hyper-parameters can be found from the `parser` in the [main.py](https://github.com/SuperBruceJia/NLNet-IQA/blob/main/main.py#L73).<br>
(3) Please change the database path `'/home/jsy/BIQA/'` to your own path.

<details>
<summary>Experimental Results</summary>
<img width="954" alt="image" src="https://user-images.githubusercontent.com/31528604/210926899-dbadfdd4-f9e5-4b78-bbeb-a637cf063e73.png">
<img width="961" alt="image" src="https://user-images.githubusercontent.com/31528604/210926960-1398dc6d-2e46-45ae-a6e5-7202b762e765.png">
</details>

### Cross-Database Evaluations
Experiments Settings:<br>
✔︎ One database is used as the training set, and the other databases are the testing sets.<br>
✔︎ The performance of the model in the last epoch (100 epochs in this work) is reported.<br>

Quick Start: (Folder: Cross Database Evaluations)<br>
```python
python cross_main.py --database_path '/home/jsy/BIQA/' --train_database TID2013 --test_database CSIQ --num_workers 8 --gpu 0
```

<details>
<summary>Experimental Results</summary>
<img width="966" alt="image" src="https://user-images.githubusercontent.com/31528604/210927394-850c1c7c-7b69-4c67-9979-231db6340923.png">
</details>

### Single Distortion Type Evaluation
Quick Start (Folder: Individual Distortion Evaluation):
```python
python TID2013-Single-Distortion.py
```
(1) Please change the trained models' path and Database path.<br>
(2) The Distortion Type of the Index can be found from original papers: [TID2013](https://www.sciencedirect.com/science/article/pii/S0923596514001490) and [KADID](http://database.mmsp-kn.de/kadid-10k-database.html). 

<details>
<summary>Experimental Results</summary>
LIVE Database:
<img width="973" alt="image" src="https://user-images.githubusercontent.com/31528604/210927080-c93f517d-fdd0-4663-8c22-3554044c8f0a.png">

---

CSIQ Database:
<img width="974" alt="image" src="https://user-images.githubusercontent.com/31528604/210927134-173b4668-ab33-4cce-8a64-16472b53c13a.png">

---

TID2013 Database:
<img width="977" alt="image" src="https://user-images.githubusercontent.com/31528604/210927196-9274be91-75c3-4481-bef1-678027016d7a.png">

---

KADID-10k Database:
<img width="967" alt="image" src="https://user-images.githubusercontent.com/31528604/210927248-c334dd50-2379-43a6-bd9b-38cf9a3810f8.png">
</details>

### Real World Image Testing
Quick Start:
```python
python real_testing.py --model_file 'save_model/TID2013-32-4-1.pth' --im_path 'test_images/cr7.jpg'
```
Please comment [these lines](https://github.com/SuperBruceJia/NLNet-IQA/blob/main/real_testing.py#L45) if you don't want to resize the original image.

## Superpixel Segmentation Demo
Quick Start (Folder: Superpixel Segmentation):
```python
python superpixel.py
```
<img width="600" alt="image" src="https://user-images.githubusercontent.com/31528604/210958870-b7037f5a-c3bf-472b-bd43-7605e02e7d00.png">

## Trained Models and Benchmark Databases
✔︎ Trained Models (Intra-Database Experiments): Download [here](https://drive.google.com/drive/folders/1K-24RGXyvSUZfnTThQ0CXUf4BgJA_pn7?usp=sharing)<br>
✔︎ Trained Models (Cross-Database Evaluations): Download [here](https://drive.google.com/drive/folders/1-9XfTt4ne057Ureecf_eLXiMQ_4xucgJ?usp=sharing)<br>
✔︎ LIVE, CSIQ, TID2013, and KADID-10k Databases: Download [here](https://drive.google.com/drive/folders/1gfBlByg1bpBXQOFZb6LyCttaX4eAf_Eh?usp=sharing)

<details>
<summary>Databases Summary</summary>
<img width="560" alt="image" src="https://user-images.githubusercontent.com/31528604/210788223-dbe067f6-64e1-4f74-a747-4c07b42088bd.png">
</details>
    
## Global Distortions and Local Distortions
**Global Distortions**: the globally and uniformly distributed distortions with non-local recurrences over the image.<br>
**Local Distortions**: the local nonuniform-distributed distortions in a local region.
    
✔︎ LIVE Database:
    
    Global Distortions: JPEG, JP2K, WN, GB
    
    Local Distortions: FF

<details>
<summary>Distortion Demo</summary>
<img width="582" alt="image" src="https://user-images.githubusercontent.com/31528604/210927999-48d2d4e2-d63a-4ece-8681-e5fbe1fb3d98.png">
</details>

✔︎ CSIQ Database:
    
    Global Distortions: JPEG, JP2K, WN, GB, PN, СС
    
    Local Distortions: There is no local distortion in CSIQ Database.

<details>
<summary>Distortion Demo</summary>
<img width="950" alt="image" src="https://user-images.githubusercontent.com/31528604/210928260-3f3d938e-53eb-43fb-90d2-7cca59850aee.png">
</details>

✔︎ TID2013 Database:
    
    Global Distortions: Additive Gaussian noise, Lossy compression of noisy images, Additive noise in color components, Comfort noise, Contrast change, Change of color saturation, Spatially correlated noise, High frequency noise, Impulse noise, Quantization noise, Gaussian blur, Image denoising, JPEG compression, JPEG 2000 compression, Multiplicative Gaussian noise, Image color quantization with dither, Sparse sampling and reconstruction, Chromatic aberrations, Masked noise, and Mean shift (intensity shift)
    
    Local Distortions: JPEG transmission errors, JPEG 2000 transmission errors, Non eccentricity pattern noise, and Local bock-wise distortions with different intensity

<details>
<summary>Distortion Demo</summary>
<img width="618" alt="image" src="https://user-images.githubusercontent.com/31528604/210928535-4f9bc8ad-f9ca-4a25-bbc0-5a9d3016a637.png">
</details>

✔︎ KADID-10k Database:
    
    Global Distortions: blurs (lens blur, motion blur, and GB), color distortions (color diffusion, color shift, color saturation 1, color saturation 2, and color quantization), compression (JPEG and JP2K), noise (impulse noise, denoise, WN, white noise in color component, and multiplicative noise), brightness change (brighten, darken, and mean shift), spatial distortions (jitter, pixelate, and quantization), and sharpness and contrast (high sharpen and contrast change)
    
    Local Distortions: Color block, and Non-eccentricity patch

<details>
<summary>Distortion Demo</summary>
<img width="970" alt="image" src="https://user-images.githubusercontent.com/31528604/210928749-1d080cc4-04b4-462e-bc3b-0e6e3344d38d.png">
</details>

## Evaluation Metrics
(1) Pearson Linear Correlation Coefficient (**PLCC**): measures the prediction accuracy<br>
(2) Spearman Rank-order Correlation Coefficient (**SRCC**): measures the prediction monotonicity<br>
A short note of the IQA evaluation metrics can be downloaded [here](https://shuyuej.com/files/MMSP/IQA_Evaluation_Metrics.pdf).

## Paper and Presentations
(1) **Original Paper** can be downloaded [here](https://shuyuej.com/files/MMSP/MMSP22_Paper.pdf).<br>
(2) **Slides Presentation** can be downloaded [here](https://shuyuej.com/files/MMSP/MMSP22_Slides.pdf).<br>
(3) **Poster Presentation** can be downloaded [here](https://shuyuej.com/files/MMSP/MMSP22_Poster.pdf).

<div>
    <div style="text-align:center">
    <img width=100%device-width src="https://github.com/SuperBruceJia/NLNet-IQA/raw/main/MMSP22_Poster.png" alt="Poster">
</div>

(i) **Image Preprocessing**: The input image is pre-processed. 👉 Check [this file](https://github.com/SuperBruceJia/NLNet-IQA/blob/main/lib/image_process.py#L17).<br>
(ii) **Graph Neural Network – Non-Local Modeling Method**: A two-stage GNN approach is presented for the non-local feature extraction and long-range dependency construction among different regions. The first stage aggregates local features inside superpixels. The following stage learns the non-local features and long-range dependencies among the graph nodes. It then integrates short- and long-range information based on an attention mechanism. The means and standard deviations of the non-local features are obtained from the graph feature signals. 👉 Check [this file](https://github.com/SuperBruceJia/NLNet-IQA/blob/main/model/network.py#L62).<br>
(iii) **Pre-trained VGGNet-16 – Local Modeling Method**: Local feature means and standard deviations are derived from the pre-trained VGGNet-16 considering the hierarchical degradation process of the HVS. 👉 Check [this file](https://github.com/SuperBruceJia/NLNet-IQA/blob/main/model/network.py#L37).<br>
(iv) **Feature Mean & Std Fusion and Quality Prediction**: The means and standard deviations of the local and non-local features are fused to deliver a robust and comprehensive representation for quality assessment. 👉 Check [this file](https://github.com/SuperBruceJia/NLNet-IQA/blob/main/model/network.py). Besides, the distortion type identification loss Lt , quality prediction loss Lq , and quality ranking loss Lr are utilized for training the NLNet. 👉 Check [this file](https://github.com/SuperBruceJia/NLNet-IQA/blob/main/model/solver.py#L171). During inference, the final quality of the image is the averaged quality of all the non-overlapping patches. 👉 Check [this file](https://github.com/SuperBruceJia/NLNet-IQA/blob/main/lib/image_process.py#L17). 

## Structure of the Code
At the root of the project, you will see:
```text
├── main.py
├── model
│   ├── layers.py
│   ├── network.py
│   └── solver.py
├── superpixel
│   └── slic.py
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
│   └── README.md
├── test_images
│   └── cr7.jpg
├── real_testing.py
```

## Citation
If you find our work useful in your research, please consider citing it in your publications. 
We provide a BibTeX entry below.

```bibtex
@INPROCEEDINGS{Jia2022NLNet,
    title     = {No-reference Image Quality Assessment via Non-local Dependency Modeling},   
    author    = {Jia, Shuvue and Chen, Baoliang and Li, Dingquan and Wang, Shiqi},  
    booktitle = {2022 IEEE 24th International Workshop on Multimedia Signal Processing (MMSP)},   
    year      = {Sept. 2022},
    volume    = {},
    number    = {},
    pages     = {01-06},
    doi       = {10.1109/MMSP55362.2022.9950035}
}
```

## Contact
If you have any question, please drop me an email at shuyuej@ieee.org.

## Acknowledgement
The authors would like to thank Dr. Xuhao Jiang, Dr. Diqi Chen, and Dr. Jupo Ma for helpful discussions and invaluable inspiration. A special appreciation should be shown to Dr. Dingquan Li because this code is built upon his [(Wa)DIQaM-FR/NR](https://github.com/lidq92/WaDIQaM) re-implementation.
