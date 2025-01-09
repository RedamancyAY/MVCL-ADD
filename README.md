# MVCL-ADD
Code for "Multi-View Collaborative Learning Network for Speech Deepfake Detection"

This repository contains the code for the paper titled "Multi-View Collaborative Learning Network for Speech Deepfake Detection" published in AAAI 2025.

## Abstract

As deep learning techniques advance rapidly, deepfake speech synthesized through text-to-speech or voice conversion networks is becoming increasingly realistic, posing significant challenges for detection and raising potential threats to social security. This growing realism has prompted extensive research in speech deepfake detection. However, current detection methods primarily focus on extracting features from either the raw waveform or the spectrogram, often overlooking the valuable correspondences between these two modalities that could enhance the detection of previously unseen types of deepfakes. In this work, we propose a multi-view collaborative learning network for speech deepfake detection, which jointly learns robust speech representations from both raw waveforms and spectrograms. 
Specifically, we first design a \textbf{D}ual-\textbf{B}ranch \textbf{C}ontrastive \textbf{L}earning (DBCL) framework for learning different view features. DBCL consists of two branches that learn representations from the raw waveform or the spectrogram and utilizes contrastive learning to enhance inter- and inner-view correlations. Additionally, we introduce a \textbf{W}aveform-\textbf{S}pectrogram \textbf{F}usion \textbf{M}odule (WSFM) to exchange multi-view information for collaborative learning. In the feature learning process, WSFM converts features between views and merges them adaptively using waveform-spectrogram cross-attention. The final detection is conducted based on the concatenation of the waveform and spectrogram features. We conduct extensive experiments on four benchmark deepfake speech detection datasets, and the experimental results demonstrate that our method can achieve better detection performance than current state-of-the-art detection methods.

## Requirements

```bash
pip install -r requirements.txt
```
Actually, the package versions are not strict. Maybe the latest versions of torch and pytorch_lightning can still work.


## Usage


One can run the following commands to train or test our multiview model.
```bash
python train.py --gpu 0 --cfg 'MultiView/ASV2019_LA'  -v 0;\
python train.py --gpu 0 --cfg 'MultiView/ASV2019_LA'  -t 1 -v 0;\

python train.py --gpu 0 --cfg 'MultiView/ASV2021_LA'  -v 0;\
python train.py --gpu 0 --cfg 'MultiView/ASV2021_LA'  -t 1 -v 0;\

python train.py --gpu 0 --cfg 'MultiView/ASV2021_inner'  -v 0;\
python train.py --gpu 0 --cfg 'MultiView/ASV2021_inner'  -t 1 -v 0;\
python train.py --gpu 0 --cfg 'MultiView/ASV2021_inner'  -t 1 -v 0 --test_noise 1 --test_noise_level 20 --test_noise_type 'bg';\

python train.py --gpu 0 --cfg 'MultiView/MLAAD_cross_lang'  -v 0;\
python train.py --gpu 0 --cfg 'MultiView/MLAAD_cross_lang'  -t 1 -v 0;\
```

## Acknowledgements

Please feel free to contact me (zkyhitsz@gmail.com) if you have any questions.

