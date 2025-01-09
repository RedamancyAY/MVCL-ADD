


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