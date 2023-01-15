# Non-local Modeling for Image Quality Assessment
<img width="1680" alt="image" src="https://user-images.githubusercontent.com/31528604/212229645-041c280b-fcaf-4d5a-8703-2d94f2fe1615.png">

## Table of Contents
<ul>
    <li><a href="#Installation">Installation</a></li>
    <li><a href="#Experiments-Settings-and-Quick-Start">Experiments Settings and Quick Start</a></li>
    <li><a href="#Superpixel-Segmentation-Demo">Superpixel Segmentation Demo</a></li>
    <li><a href="#Trained-Models-and-Benchmark-Databases">[Download] Trained Models and Benchmark Databases</a></li>
    <li><a href="#Evaluation-Metrics">Evaluation Metrics</a></li>
    <li><a href="#Motivation">Motivation</a></li>
    <li><a href="#Local-Modeling-and-Non-local-Modeling">[Definition] Local Modeling and Non-local Modeling</a></li>
    <li><a href="#Global-Distortions-and-Local-Distortions">[Definition] Global Distortions and Local Distortions</a></li>
    <li><a href="#Paper-and-Presentations">[Download] Paper and Presentations</a></li>
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
(1) Other hyper-parameters can also be modified via `--parameter XXX`, _e.g._, `--epochs 200` and `--lr 1e-5`.<br>
(2) Hyper-parameters can be found from the `parser` in the [main.py](https://github.com/SuperBruceJia/NLNet-IQA/blob/main/main.py#L73).<br>
(3) Please change the database path `'/home/jsy/BIQA/'` to your own path.

<details>
<summary>Experimental Results</summary>
<img width="900" alt="image" src="https://user-images.githubusercontent.com/31528604/211454477-1f112208-6f3f-45fe-8cfc-86fb311e243a.png">
<img width="900" alt="image" src="https://user-images.githubusercontent.com/31528604/211454790-904f12cb-ae83-4bb2-8eea-bd762b64c0f4.png">
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
<img width="900" alt="image" src="https://user-images.githubusercontent.com/31528604/211455285-b29db97a-29d8-499a-a728-5707afe56e22.png">
</details>

### Single Distortion Type Evaluation
Quick Start (Folder: Individual Distortion Evaluation):
```python
python TID2013-Single-Distortion.py
```
(1) Please change the trained models' path and Database path.<br>
(2) The Index of Distortion Type can be found from original papers: [TID2013](https://www.sciencedirect.com/science/article/pii/S0923596514001490) and [KADID](http://database.mmsp-kn.de/kadid-10k-database.html#:~:text=blurs). 

<details>
<summary>Experimental Results</summary>

LIVE Database:

<img width="700" alt="image" src="https://user-images.githubusercontent.com/31528604/211454955-d9346292-b718-45f5-8f8a-14c81cc19586.png">

---

CSIQ Database:

<img width="700" alt="image" src="https://user-images.githubusercontent.com/31528604/211455036-99a31158-967d-46b4-8ba1-4a2187447373.png">

---

TID2013 Database:

<img width="1400" alt="image" src="https://user-images.githubusercontent.com/31528604/211455110-c48a94ca-599c-45a5-97e7-4d735cd994e5.png">

---

KADID-10k Database:

<img width="1400" alt="image" src="https://user-images.githubusercontent.com/31528604/211455189-c367264c-03c5-49d0-8388-e8cdb1de6a49.png">
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

<details>
<summary>Superpixel vs. Square Patch Representation Demo</summary>
<img width="900" alt="image" src="https://user-images.githubusercontent.com/31528604/210959208-6381e2f1-0b0f-4bd6-90b2-8a2039c08a09.png">
<img width="1600" alt="image" src="https://user-images.githubusercontent.com/31528604/212520023-5d774076-d10a-4cdc-b773-0115e4bd1c81.png">
<img width="1600" alt="image" src="https://user-images.githubusercontent.com/31528604/211976748-3cbae528-d4e8-4f7a-893f-f04110e36abe.png">
<img width="1600" alt="image" src="https://user-images.githubusercontent.com/31528604/211976777-e617f93f-9237-407d-bdcf-36874f4c66d4.png">
</details>

## Trained Models and Benchmark Databases
✔︎ Trained Models (Intra-Database Experiments): Download [here](https://drive.google.com/drive/folders/1K-24RGXyvSUZfnTThQ0CXUf4BgJA_pn7?usp=sharing)<br>
✔︎ Trained Models (Cross-Database Evaluations): Download [here](https://drive.google.com/drive/folders/1-9XfTt4ne057Ureecf_eLXiMQ_4xucgJ?usp=sharing)<br>
✔︎ LIVE, CSIQ, TID2013, and KADID-10k Databases: Download [here](https://drive.google.com/drive/folders/1gfBlByg1bpBXQOFZb6LyCttaX4eAf_Eh?usp=sharing)

<details>
<summary>Databases Summary</summary>
<img width="900" alt="image" src="https://user-images.githubusercontent.com/31528604/211455700-1436735c-eec6-4670-b509-2bf784a11aee.png">
<img width="900" alt="image" src="https://user-images.githubusercontent.com/31528604/211731935-e1559cf9-15aa-4a33-bf86-540deb70028a.png">
</details>

## Evaluation Metrics
(1) Pearson Linear Correlation Coefficient (**PLCC**): measures the prediction accuracy<br>
(2) Spearman Rank-order Correlation Coefficient (**SRCC**): measures the prediction monotonicity<br>
✔︎ A short note of the IQA evaluation metrics can be downloaded [here](https://shuyuej.com/files/MMSP/IQA_Evaluation_Metrics.pdf).<br>
✔︎ In the [code](https://github.com/SuperBruceJia/NLNet-IQA/blob/main/lib/utils.py#L29) (`evaluation_criteria` function), PLCC, SRCC, Kendall Rank-order Correlation Coefficient (KRCC), Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and Outlier Ratio (OR) are all calculated. In this work, I only compare the PLCC and SRCC among different IQA algorithms.

## Motivation
**Local Content**: HVS is adaptive to the local content.<br>
**Long-range Dependency and Relational Modeling**: HVS perceives image quality with long-range dependency constructed among different regions.

<img width="1460" alt="image" src="https://user-images.githubusercontent.com/31528604/212519878-0da16724-750d-43a2-a083-fc593463ad43.png">
<img width="1460" alt="image" src="https://user-images.githubusercontent.com/31528604/212519886-c7fb7ca1-ec50-4b7d-8e21-2287e00cf29c.png">

## Local Modeling and Non-local Modeling
**Local Modeling**: The local modeling methods encode spatially proximate local neighborhoods.<br>
**Non-local Modeling**: The non-local modeling establishes the spatial integration of information by long- and short-range communications with different spatial weighting functions.

<img width="1460" alt="image" src="https://user-images.githubusercontent.com/31528604/212520317-88da5cb5-44be-41d5-bcbe-1632c2c35811.png">
<img width="1460" alt="image" src="https://user-images.githubusercontent.com/31528604/212519914-d3948de4-dc2e-4125-9414-9725aa76af03.png">

<details>
<summary>Non-local Behavior Demo</summary>
<img width="900" alt="image" src="https://user-images.githubusercontent.com/31528604/211026397-7990fdbd-b41a-414a-a40f-ec4ecb637dcf.png">
<img width="900" alt="image" src="https://user-images.githubusercontent.com/31528604/211026979-60e49649-75c5-481f-86d0-021c2ad5cde6.png">
</details>

<details>
<summary>Local Modeling vs. Non-local Modeling Demo</summary>
<img width="900" alt="image" src="https://user-images.githubusercontent.com/31528604/211028273-373e8139-f111-40be-b214-019a64b90392.png">
</details>

## Global Distortions and Local Distortions
**Global Distortions**: the globally and uniformly distributed distortions with non-local recurrences over the image.<br>
**Local Distortions**: the local nonuniform-distributed distortions in a local region.

<img width="1460" alt="image" src="https://user-images.githubusercontent.com/31528604/212519962-20781700-29a6-4a2f-97a9-13beaaf8ac72.png">

<img width="1460" alt="image" src="https://user-images.githubusercontent.com/31528604/212447958-f8011613-e26b-4bf4-993a-b56c395703b6.png">
✔︎ LIVE Database:
    
    Global Distortions: JPEG, JP2K, WN, and GB
    
    Local Distortions: FF

<details>
<summary>Distortion Demo</summary>
<img width="900" alt="image" src="https://user-images.githubusercontent.com/31528604/210927999-48d2d4e2-d63a-4ece-8681-e5fbe1fb3d98.png">
</details>

✔︎ CSIQ Database:
    
    Global Distortions: JPEG, JP2K, WN, GB, PN, and СС
    
    Local Distortions: There is no local distortion in CSIQ Database.

<details>
<summary>Distortion Demo</summary>
<img width="900" alt="image" src="https://user-images.githubusercontent.com/31528604/210928260-3f3d938e-53eb-43fb-90d2-7cca59850aee.png">
</details>

✔︎ TID2013 Database:
    
    Global Distortions: Additive Gaussian noise, Lossy compression of noisy images, Additive noise in color components, Comfort noise, Contrast change, Change of color saturation, Spatially correlated noise, High frequency noise, Impulse noise, Quantization noise, Gaussian blur, Image denoising, JPEG compression, JPEG 2000 compression, Multiplicative Gaussian noise, Image color quantization with dither, Sparse sampling and reconstruction, Chromatic aberrations, Masked noise, and Mean shift (intensity shift)
    
    Local Distortions: JPEG transmission errors, JPEG 2000 transmission errors, Non eccentricity pattern noise, and Local bock-wise distortions with different intensity

<details>
<summary>Distortion Demo</summary>
<img width="900" alt="image" src="https://user-images.githubusercontent.com/31528604/210928535-4f9bc8ad-f9ca-4a25-bbc0-5a9d3016a637.png">
</details>

✔︎ KADID-10k Database:
    
    Global Distortions: blurs (lens blur, motion blur, and GB), color distortions (color diffusion, color shift, color saturation 1, color saturation 2, and color quantization), compression (JPEG and JP2K), noise (impulse noise, denoise, WN, white noise in color component, and multiplicative noise), brightness change (brighten, darken, and mean shift), spatial distortions (jitter, pixelate, and quantization), and sharpness and contrast (high sharpen and contrast change)
    
    Local Distortions: Color block and Non-eccentricity patch

<details>
<summary>Distortion Demo</summary>
<img width="900" alt="image" src="https://user-images.githubusercontent.com/31528604/210928749-1d080cc4-04b4-462e-bc3b-0e6e3344d38d.png">
</details>

## Paper and Presentations
(1) **Original Paper** can be downloaded [here](https://shuyuej.com/files/MMSP/MMSP22_Paper.pdf).<br>
(2) **Detailed Slides Presentation** can be downloaded [here](https://shuyuej.com/files/Presentation/A_Summary_Three_Projects.pdf).<br>
(3) **Detailed Slides Presentation with Animations** can be downloaded [here](https://shuyuej.com/files/Presentation/A_Summary_Three_Projects_Animations.pdf).<br>
(4) **Simple Slides Presentation** can be downloaded [here](https://shuyuej.com/files/MMSP/MMSP22_Slides.pdf).<br>
(5) **Poster Presentation** can be downloaded [here](https://shuyuej.com/files/MMSP/MMSP22_Poster.pdf).

### Model Overiew
<div>
    <div style="text-align:center">
    <img width=100%device-width src="https://github.com/SuperBruceJia/NLNet-IQA/raw/main/overview.png" alt="NLNet">
</div>

(i) **Image Preprocessing**: The input image is pre-processed. 👉 Check [this file](https://github.com/SuperBruceJia/NLNet-IQA/blob/main/lib/image_process.py#L17).<br>
(ii) **Graph Neural Network – Non-Local Modeling Method**: A two-stage GNN approach is presented for the non-local feature extraction and long-range dependency construction among different regions. The first stage aggregates local features inside superpixels. The following stage learns the non-local features and long-range dependencies among the graph nodes. It then integrates short- and long-range information based on an attention mechanism. The means and standard deviations of the non-local features are obtained from the graph feature signals. 👉 Check [this file](https://github.com/SuperBruceJia/NLNet-IQA/blob/main/model/network.py#L62).<br>
(iii) **Pre-trained VGGNet-16 – Local Modeling Method**: Local feature means and standard deviations are derived from the pre-trained VGGNet-16 considering the hierarchical degradation process of the HVS. 👉 Check [this file](https://github.com/SuperBruceJia/NLNet-IQA/blob/main/model/network.py#L37).<br>
(iv) **Feature Mean & Std Fusion and Quality Prediction**: The means and standard deviations of the local and non-local features are fused to deliver a robust and comprehensive representation for quality assessment. 👉 Check [this file](https://github.com/SuperBruceJia/NLNet-IQA/blob/main/model/network.py). Besides, the distortion type identification loss $L_t$ , quality prediction loss $L_q$ , and quality ranking loss $L_r$ are utilized for training the NLNet. 👉 Check [this file](https://github.com/SuperBruceJia/NLNet-IQA/blob/main/model/solver.py#L171). During inference, the final quality of the image is the averaged quality of all the non-overlapping patches. 👉 Check [this file](https://github.com/SuperBruceJia/NLNet-IQA/blob/main/lib/image_process.py#L17). 

### Poster Presentation
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
@inproceedings{Jia2022NLNet,
    title     = {No-reference Image Quality Assessment via Non-local Dependency Modeling},   
    author    = {Jia, Shuyue and Chen, Baoliang and Li, Dingquan and Wang, Shiqi},  
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
