## 3rd place solution for HubMap 2023 Challenge hosted on Kaggle

This documentation outlines how to reproduce the 3rd place solution for the instance segmentation competition on Kaggle hosted by HubMap.


## Preprocessing
You can download the preprocessed data (coco annotations) directly using this command `kaggle datasets download -d nischaydnk/hubmap-coco-datasets`. Nevertheless the code for converting the raw images and masks into coco annotations has been provided in `kfoldgen.ipynb` in split folder.


## Reproduce solution 

For training and replicating our final solution, we have added python scripts for each models in separate folders. There might be some manual work required to reproduce the training results (ex. changing the path to best model in previous stage)
- [Stage 1 without Pseudo](https://github.com/Nischaydnk/HubMap-2023-3rd-Place-Solution/tree/main/all_configs/nops_config_pret)
- [Stage 2 without Pseudo](https://github.com/Nischaydnk/HubMap-2023-3rd-Place-Solution/tree/main/all_configs/nops_config_finetune)
- [Stage 1 with Pseudo](https://github.com/Nischaydnk/HubMap-2023-3rd-Place-Solution/tree/main/all_configs/pseudo_config_pret)
- [Stage 2 with Pseudo](https://github.com/Nischaydnk/HubMap-2023-3rd-Place-Solution/tree/main/all_configs/pseudo_config_finetune)

#### Note: Along with .py config files, I have attached the logs for each experiments alongside with extension .log, that would be helpful to compare the results while reproducing the solution.

For inference notebooks and model weights, you may visit our final submission [notebook](https://www.kaggle.com/code/nischaydnk/cv-wala-mega-ensemble-hubmap-2023)


## Hardware

All of the single models were trained using 3x A6000(local) instances with GPU enabled to run all data preprocessing, model training, and inference was done with kaggle notebooks. 

[https://www.kaggle.com/docs/notebooks](https://www.kaggle.com/docs/notebooks)

## Software

We used [Kaggle GPU notebooks](https://github.com/Kaggle/docker-python/blob/master/gpu.Dockerfile) to run all our inference scripts.

Below are the packages used in addition to the ones included in the default train scripts provided. All packages were installed via uploaded kaggle dataset. Please make sure to install the following packages in similar order so that you don't face dependencies issues.

| Package Name | Repository | Kaggle Dataset |
| --- |--- | --- |
| pytorch | https://github.com/pytorch/pytorch | |
| albumentations | https://github.com/albumentations-team/albumentations |  |
| pytorch image models | https://github.com/rwightman/pytorch-image-models | https://www.kaggle.com/benihime91/pytorch-image-models |
| pycocotools |https://pypi.org/project/pycocotools/| https://www.kaggle.com/datasets/itsuki9180/pycocotools-206 |
| addict |https://www.kaggle.com/datasets/isps737/mmdetection-2-26-0| https://www.kaggle.com/datasets/isps737/mmdetection-2-26-0 |
| yapf | https://www.kaggle.com/datasets/isps737/mmdetection-2-26-0 |https://www.kaggle.com/datasets/isps737/mmdetection-2-26-0 |
| terminal | https://www.kaggle.com/datasets/isps737/mmdetection-2-26-0 |https://www.kaggle.com/datasets/isps737/mmdetection-2-26-0 |
| terminal tables | https://www.kaggle.com/datasets/isps737/mmdetection-2-26-0 |https://www.kaggle.com/datasets/isps737/mmdetection-2-26-0 |
| mmcv_full=1.17.1 | https://pypi.org/project/mmcv-full/ | https://www.kaggle.com/code/atom1231/mmdet3-wheels |
| ensemble-boxes=1.0.4| https://github.com/ZFTurbo/Weighted-Boxes-Fusion | https://www.kaggle.com/vgarshin/ensemble-boxes-104 |
| cbnetv2 | https://github.com/VDIGPKU/CBNetV2 | https://www.kaggle.com/datasets/nischaydnk/cbnetv2-repo|
| MMdetection=2.26.0 | https://github.com/open-mmlab/mmdetection | https://www.kaggle.com/datasets/isps737/mmdetection-2-26-0|
| instaboostfast | https://pypi.org/project/instaboostfast/ | |
| vit adapter | https://github.com/czczup/ViT-Adapter | https://www.kaggle.com/datasets/nischaydnk/vitadadapter |




## Data Used

Please download the following datasets from the Kaggle to make things easier to train. 

`kaggle datasets download -d nischaydnk/hubmap-coco-datasets`

`kaggle datasets download -d nischaydnk/hubmap-coco-pretrained-models`

`kaggle competitions download -c hubmap-hacking-the-human-vasculature`

After unzip. Ideally, coco datasets should be accessed with coco_data folder, hubmap competition train data with train folder, and pretrained models from main folder itself. You can always change the paths in python config files, it's quite straightforward. 

To run multi-gpu training, you will need to update the config file path in the dist_train.sh for running certain experiments. 

## Models Training 

Once all the datasets are downloaded and unzipped. You can training each of the models in following steps: 
- Change the Config file path for stage 1 config in dist_train.sh
- Run `bash dist_train.sh`
- Change the load_from parameter in stage 2 models based on the best saved epoch.
- Change the Config file path for stage 1 config in dist_train.sh
- Run `bash dist_train.sh`

### Non-Pseudo label models:

#### Detectors Resnext 101 based model

- Change path parameter in dist_train.sh to `all_configs/nops_config_pret/htc_resnext101_exp1pret_wsiall.py`
- `bash dist_train.sh`
- Select path of best saved epoch from work_dir folder selected in stage 1 config.
- Change load_from parameter in `all_configs/nops_config_finetune/pretwsiallhtc_resnext101_exp3_augv4_maskloss4.py`
- Change path parameter in dist_train.sh to `all_configs/nops_config_finetune/pretwsiallhtc_resnext101_exp3_augv4_maskloss4.py`
- `bash dist_train.sh` 


#### CBNetV2 Base model

- Change path parameter in dist_train.sh to `all_configs/nops_config_pret/pretexp1_cbnetbase_1600_10e.py`
- `bash dist_train.sh`
- Select path of last saved epoch from work_dir folder selected in stage 1 config.
- Change load_from parameter in `all_configs/nops_config_finetune/exp1_withpret_cbbase_1600_morepretep.py`
- Change path parameter in dist_train.sh to `all_configs/nops_config_finetune/exp1_withpret_cbbase_1600_morepretep.py`
- `bash dist_train.sh`


### Generate Pseudo Labels on dataset 3

#### Vit Adapter Beit v2l htc++ w/ SGD (no pseudo)

- Change path parameter in dist_train.sh to `all_configs/pseudo_config_pret/pretexp3_adaplargebeitv2l_htc.py`
- `bash dist_train.sh`
- Select path of best saved epoch from work_dir folder selected in stage 1 config.
- Change load_from parameter in `all_configs/pseudo_config_finetune/pretexp4_adapbeitv2lhtc_1400_ds2wsiall_ps60exp2_lossw2_adamp2_leak.py`
- Change path parameter in dist_train.sh to `all_configs/pseudo_config_finetune/pretexp4_adapbeitv2lhtc_1400_ds2wsiall_ps60exp2_lossw2_adamp2_leak.py`
- `bash dist_train.sh`


Once No pseudo Resnext 101, CbnetV2 Base and Vit adapter Beit v2l models are trained on multistage pipeline, use those weights to perform inference in `split/pseudogen.ipynb' for generating pseudo labels. You may save the pseudo labels in whatever path you want, but will need to change the data path in config files accordingly. 

  
### Pseudo label models:

#### Vit Adapter Beit v2l htc++ w/ AdamW

- Change path parameter in dist_train.sh to `all_configs/pseudo_config_pret/pretexp3_adaplargebeitv2l_htc.py`
- `bash dist_train.sh`
- Select path of best saved epoch from work_dir folder selected in stage 1 config.
- Change load_from parameter in `all_configs/pseudo_config_finetune/pretexp4_adapbeitv2lhtc_1400_ds2wsiall_ps60exp2_lossw2_adamp2_leak.py`
- Change path parameter in dist_train.sh to `all_configs/pseudo_config_finetune/pretexp4_adapbeitv2lhtc_1400_ds2wsiall_ps60exp2_lossw2_adamp2_leak.py`
- `bash dist_train.sh` 


#### Vit Adapter Beit v2l htc++ w/ SGD

- Change path parameter in dist_train.sh to `all_configs/pseudo_config_pret/pretexp1_adaplargebeitv2l_htc.py`
- `bash dist_train.sh`
- Select path of best saved epoch from work_dir folder selected in stage 1 config.
- Change load_from parameter in `all_configs/pseudo_config_finetune/exp4_adapbeitv2l_withps50exp2.py`
- Change path parameter in dist_train.sh to `all_configs/pseudo_config_finetune/exp4_adapbeitv2l_withps50exp2.py`
- `bash dist_train.sh`


#### Detectors based Resnet 50

- Change path parameter in dist_train.sh to `all_configs/pseudo_config_pret/ds2allwsiprethtc50ps60.py`
- `bash dist_train.sh`
- Select path of best saved epoch from work_dir folder selected in stage 1 config.
- Change load_from parameter in `all_configs/pseudo_config_finetune/ds1pretexp1moreaug-htc50-2048-cv408ps.py`
- Change path parameter in dist_train.sh to `all_configs/pseudo_config_finetune/ds1pretexp1moreaug-htc50-2048-cv408ps.py`
- `bash dist_train.sh`


Once all the models are trained, you may use them for inference. The names of config files used in the Stage 2 are same as the dataset name hosted on Kaggle for performing final submission.

