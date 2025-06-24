# MVCL-ADD
Code for "Multi-View Collaborative Learning Network for Speech Deepfake Detection"

This repository contains the code for the paper titled "Multi-View Collaborative Learning Network for Speech Deepfake Detection" published in AAAI 2025.

## Abstract

As deep learning techniques advance rapidly, deepfake speech synthesized through text-to-speech or voice conversion networks is becoming increasingly realistic, posing significant challenges for detection and raising potential threats to social security. This growing realism has prompted extensive research in speech deepfake detection. However, current detection methods primarily focus on extracting features from either the raw waveform or the spectrogram, often overlooking the valuable correspondences between these two modalities that could enhance the detection of previously unseen types of deepfakes. In this work, we propose a multi-view collaborative learning network for speech deepfake detection, which jointly learns robust speech representations from both raw waveforms and spectrograms. 
Specifically, we first design a \textbf{D}ual-\textbf{B}ranch \textbf{C}ontrastive \textbf{L}earning (DBCL) framework for learning different view features. DBCL consists of two branches that learn representations from the raw waveform or the spectrogram and utilizes contrastive learning to enhance inter- and inner-view correlations. Additionally, we introduce a \textbf{W}aveform-\textbf{S}pectrogram \textbf{F}usion \textbf{M}odule (WSFM) to exchange multi-view information for collaborative learning. In the feature learning process, WSFM converts features between views and merges them adaptively using waveform-spectrogram cross-attention. The final detection is conducted based on the concatenation of the waveform and spectrogram features. We conduct extensive experiments on four benchmark deepfake speech detection datasets, and the experimental results demonstrate that our method can achieve better detection performance than current state-of-the-art detection methods.

## Requirements

You can create a new conda env to run this model.
```bash
conda create -n mvcl python=3.9
conda activate mvcl
```

Install packages:
```bash
pip install torch torchaudio pytorch_lightning pandas librosa hide_warnings einops transformers torchvision matplotlib rich wave
```
Actually, the package versions are not strict. Maybe the latest versions of `torch` and `pytorch_lightning` can still work.

In my personal machine, my device information is:
- RTX 4090 GPU
- CUDA version: 12.2
- NVIDIA driver version: 535.230.02

and I push the package versions of my environment in the `requirements.txt` for reference.



## Usage

You can find the model demo usage in the `demo.ipynb`. You have to integrate it into your own project: load your own dataset and build your own dataloaders. 


### scripts

~~One can run the following commands to train or test our multiview model.~~

Warning!!! don't use `train.py` since the full scripts to load the dataset, build dataloaders and train/test models are not complete. I will upload the full scripts within one week.



## Acknowledgements

Please feel free to contact me (zkyhitsz@gmail.com) if you have any questions.

Please cite the following paper if you use our method:
```bibtex
@inproceedings{zhangMultiViewCollaborativeLearning2025,
  title = {Multi-{{View Collaborative Learning Network}} for {{Speech Deepfake Detection}}},
  booktitle = {Proceedings of the {{AAAI Conference}} on {{Artificial Intelligence}}},
  author = {Zhang, Kuiyuan and Hua, Zhongyun and Lan, Rushi and Guo, Yifang and Zhang, Yushu and Xu, Guoai},
  year = {2025},
  month = apr,
  volume = {39},
  pages = {1075--1083},
  doi = {10.1609/aaai.v39i1.32094},
}

```

