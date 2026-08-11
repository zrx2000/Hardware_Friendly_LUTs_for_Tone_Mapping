# Hardware_Friendly_LUTs_for_Tone_Mapping
This is a codebase that implements the algorithm in the paper *__Learning Hardware-Friendly Lookup Tables for Tone Mapping__* published in IEEE TCSVT.

## Datasets and Model Weights
In our paper, we use the 480p 16-bit images in Google HDR+ dataset and MIT-Adobe FiveK dataset. The dataset partitioning is the same as [Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).
Since the whole dataset is too large, I only provide the Google HDR+ dataset and our results on the two test sets [here](https://drive.google.com/drive/folders/13EF-mpDu9MZ51nLsI6uVkZL7IfwFsWPv?usp=sharing).
If you would like to retrain the model, you can download the whole datasets following the guidance in this codebase: [Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).
After downloading the images, please put them in `datasets/`. 

Our model weights are put in `saved_models/`.

## Test
You can run

`python eval_hdr_plus.py` 

and 

`python eval_fivek.py` 

to test our pre-trained models on the test sets of Google HDR+ dataset and MIT-Adobe FiveK dataset respectively.

You can also directly download our results [here](https://drive.google.com/drive/folders/13EF-mpDu9MZ51nLsI6uVkZL7IfwFsWPv?usp=sharing).

## Train
`python train.py`

This script (`train.py`) trains the model using the Google HDR+ dataset by default. Before training, please write the filenames of the training set and the validation set into `train.txt` and `val.txt` respectively.

## Acknowledgement

Some scripts are based on this project: [Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT), which inspires our work.