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

Below are the packages used in addition to the ones included in the default train scripts provided. All packages were installed via uploaded kaggle dataset.

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



