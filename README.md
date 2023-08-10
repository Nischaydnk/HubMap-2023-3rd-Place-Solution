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

- Change path parameter in dist_train.sh to `all_configs/pretconf/pretexp1_adaplargebeitv2l_htc-Copy1.py`
- `bash dist_train.sh`
- Select path of best saved epoch from work_dir folder selected in stage 1 config.
- Change load_from parameter in `all_configs/nops_config_finetune/exp4_adapbeitv2l.py`
- Change path parameter in dist_train.sh to `all_configs/nops_config_finetune/exp4_adapbeitv2l.py`
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


## Overview

**Winning solution consists of :**

**5** MMdet based models with different architectures.
**2x ViT-Adapter-L** (https://github.com/czczup/ViT-Adapter/tree/main/detection)
**1x CBNetV2 Base** (https://github.com/VDIGPKU/CBNetV2)
**1x Detectors ResNeXt-101-32x4d** (https://github.com/joe-siyuan-qiao/DetectoRS)
**1x Detectors Resnet 50** 

I also had few Vit Adapter based single models which could have placed me on 1st/2nd ranks but I didn't select. No regrets :))

## Image Data Used

I only used competition for training models. *No external image data was used.* 

## How to use dataset 2??

Making the best use of dataset 2 was one of the key things to figure out in the competition. For me multi stage approach turned to be giving the highest boost. Basically during stage 1, A coco pretrained model will be loaded and pretrained on all the WSIs present in noisy annotations (dataset 2) for less epochs (~10), using really high learning rate (0.02+), with a cosine scheduler with minimum lr around (0.01), light augmentations.  

In stage 2, we will load the pretained model from stage 1 and fine-tune it on dataset 1 with higher number of epochs (15-25), heavy augmentations, higher image resolution (for some models), slightly lower starting learning rate and minimum LR till 1e-7. 


![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4712534%2Fdfbae6a6f634c755d9ea39ca2daa9751%2FScreenshot%202023-08-09%20at%203.36.59%20AM.png?generation=1691532480119818&alt=media)

*I have used Pseudo labels using dataset 3 in training few models of final ensemble solution, although I didn't find any boost using them in the leaderboard scores, I will still talk about it as they were used in final solution*


![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4712534%2Facf9ee081989a328c9f150458b4fce95%2FScreenshot%202023-08-09%20at%204.01.34%20AM.png?generation=1691533922010932&alt=media)


Using multistage approach gave around consistent 2-3% boost on validation and 4-6% improvement in leaderboard scores which is quite huge. The gap between cv & lb scores was quite obvious as models were now learning on WSI 3 & 4. Also, I never had to worry about using dilation or not as my later stage was just fine-tuned on dataset 1 (noise free annotations), so dilation doesn't help if applied directly on the masks.

# Models Summary

## Vit Adapter: These models were published in recent ICLR 2023 & turned out to be highest scoring architectures. 
- Pretrained coco weights were used. 
- 1400 x 1400 Image size (dataset fold-1 with pseudo threshold 0.5) & (full data dataset 1 with pseudo threshold 0.6)
- Loss fnc: Mask Head loss multiplied by 2x in decoder.
- 1200 x 1200 Image size used in stage 1.
- Cosine Scheduler with warmup were used.
- SGD optimizer for fold 1 model & AdamW for full data model
- Higher Image Size + Multi Scale Inference (1600x1600, 1400x1400)

**Best Public Leaderboard single model: 0.600**
**Best Private Leaderboard single model: 0.589**


##CBNetV2: Another popular set of architectures based on Swin transformers. 
- Pretrained coco weights were used. 
- 1600 x 1600 Image size (dataset 1 fold-5 without Pseudo)
- 1400 x 1400 Image size used in stage 1.
- Cosine Scheduler with warmup were used.
- Higher Image Size during Inference (2048x2048)
- SGD optimizer 

**Best Public Leaderboard single model: 0.567**

## Detectors HTC based models:  CNN based encoders for more diversity
- Pretrained coco weights were used. 
- 2048 x 2048 image size (Resnet50 fold 1 w/ pseudo threshold 0.5 , Resnext101d without pseudo)
- Loss fnc: Mask Head loss 4x for Resnext101, 1x for Resnet50 
- Cosine Scheduler with warmup were used.
- SGD optimizer 

**Public Leaderboard single model: 0.573 ( resnext 101) , 0.558 (resnet50)**


##**Techniques which provided consistent boost:**

  1. Multi Stage Training
  2. Flip based Test Time Augmentation
  3. Higher weights to Mask head in HTC based models
  4. SGD optimizer 
  5. Weighted Box Fusion for Ensemble
  6. Post Processing


## Post Processing & Ensemble

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4712534%2F0fb5299755c5bfa0d6da43263a8be223%2FScreenshot%202023-08-09%20at%204.38.49%20AM.png?generation=1691536182102653&alt=media)

As mentioned earlier, I used WBF to do ensemble. To increase the diversity, I kept NMS for TTA and WBF for ensemble. Also, using both CNN / Transformer based encoders helped in increasing higher diversity and hence more impactful ensemble. 

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4712534%2F764ea7995470a76b07549107c4b531a2%2FScreenshot%202023-08-09%20at%204.26.47%20AM.png?generation=1691536206638981&alt=media)

After the ensemble, I think some of my mask predictions got a little distorted. Therefore, I applied erosion followed by single iteration of dilation. This Post-processing gave me a decent amount of boost in both cross validation score as well as on leaderboard (+ 0.005)
 

##Light Augmentations

```
dict(
    type='RandomFlip',
    direction=['horizontal', 'vertical'],
    flip_ratio=0.5),
dict(
    type='AutoAugment',
    policies=[[{
        'type': 'Shear',
        'prob': 0.4,
        'level': 0
    }], [{
        'type': 'Translate',
        'prob': 0.4,
        'level': 5
    }],
              [{
                  'type': 'ColorTransform',
                  'prob': 1.0,
                  'level': 6
              }, {
                  'type': 'EqualizeTransform'
              }]]),
dict(
    type='Albu',
    transforms=[
        dict(
            type='ShiftScaleRotate',
            shift_limit=0.0625,
            scale_limit=0.15,
            rotate_limit=15,
            p=0.4)
    ],
    bbox_params=dict(
        type='BboxParams',
        format='pascal_voc',
        label_fields=['gt_labels'],
        min_visibility=0.0,
        filter_lost_elements=True)
```


## Heavy Augmentations

```
dict(
                type='RandomFlip',
                direction=['horizontal', 'vertical'],
                flip_ratio=0.5),
            dict(
                type='AutoAugment',
                policies=[[{
                    'type': 'Shear',
                    'prob': 0.4,
                    'level': 0
                }], [{
                    'type': 'Translate',
                    'prob': 0.4,
                    'level': 5
                }],
                          [{
                              'type': 'ColorTransform',
                              'prob': 0.6,
                              'level': 10
                          }, {
                              'type': 'BrightnessTransform',
                              'prob': 0.6,
                              'level': 3
                          }],
                          [{
                              'type': 'ColorTransform',
                              'prob': 0.6,
                              'level': 10
                          }, {
                              'type': 'ContrastTransform',
                              'prob': 0.6,
                              'level': 5
                          }],
                          [{
                              'type': 'PhotoMetricDistortion',
                              'brightness_delta': 32,
                              'contrast_range': (0.5, 1.5),
                              'hue_delta': 15
                          }],
                          [{
                              'type': 'MinIoURandomCrop',
                              'min_ious': (0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                              'min_crop_size': 0.2
                          }],
                          [{
                              'type':
                              'CutOut',
                              'n_holes': (3, 8),
                              'cutout_shape': [(4, 4), (4, 8), (8, 4), (8, 8),
                                               (16, 32), (32, 16), (32, 32),
                                               (32, 48), (48, 32), (48, 48)]
                          }],
                          [{
                              'type': 'EqualizeTransform',
                              'prob': 0.6
                          }, {
                              'type': 'BrightnessTransform',
                              'prob': 0.6,
                              'level': 3
                          }],
                          [{
                              'type': 'PhotoMetricDistortion',
                              'brightness_delta': 32,
                              'contrast_range': (0.5, 1.5),
                              'hue_delta': 18
                          }],
                          [{
                              'type': 'MinIoURandomCrop',
                              'min_ious': (0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                              'min_crop_size': 0.3
                          }],
                          [{
                              'type':
                              'CutOut',
                              'n_holes': (5, 10),
                              'cutout_shape': [(4, 4), (4, 8), (8, 4), (8, 8),
                                               (16, 32), (32, 16), (32, 32),
                                               (32, 48), (48, 32), (48, 48)]
                          }],
                          [{
                              'type': 'BrightnessTransform',
                              'prob': 0.6,
                              'level': 4
                          }, {
                              'type': 'ContrastTransform',
                              'prob': 0.6,
                              'level': 6
                          }, {
                              'type': 'Rotate',
                              'prob': 0.6,
                              'level': 10
                          }],
                          [{
                              'type': 'ColorTransform',
                              'prob': 1.0,
                              'level': 6
                          }, {
                              'type': 'EqualizeTransform'
                          }]]),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='ShiftScaleRotate',
                        shift_limit=0.0625,
                        scale_limit=0.15,
                        rotate_limit=15,
                        p=0.5),
                    dict(type='RandomRotate90', p=0.5),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='ElasticTransform',
                                alpha=120,
                                sigma=6.0,
                                alpha_affine=3.5999999999999996,
                                p=1),
                            dict(type='GridDistortion', p=1),
                            dict(
                                type='OpticalDistortion',
                                distort_limit=2,
                                shift_limit=0.5,
                                p=1)
                        ],
                        p=0.3)
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_labels'],
                    min_visibility=0.0,
                    filter_lost_elements=True) 
```


Thank you all, I've tried my best to cover most part of my solution. Again, I am super happy to win the solo gold, feel free to reach out in case you find difficulty understanding any part of it.






