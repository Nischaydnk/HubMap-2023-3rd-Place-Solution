num_things_classes = 1
num_classes = num_things_classes 
# pretrained = 'https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21k.pth'
# pretrained = 'pretrained/beitv2_large_patch16_224_pt1k_ft21k.pth'
model = dict(
    type='Mask2Former',
    backbone=dict(
        type='BEiTAdapter',
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        init_values=1e-6,
        drop_path_rate=0.4,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        window_attn=[True, True, True, True, True, True,
                     True, True, True, True, True, True,
                     True, True, True, True, True, True,
                     True, True, True, True, True, True],
        window_size=[14, 14, 14, 14, 14, 56,
                     14, 14, 14, 14, 14, 56,
                     14, 14, 14, 14, 14, 56,
                     14, 14, 14, 14, 14, 56],
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        pretrained=False),
  

    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=2.0),
            mask_cost=dict(
                type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
            dice_cost=dict(
                type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=False,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=True,
        # max_per_image is for instance segmentation.
        max_per_image=100,
        iou_thr=0.8,
        # In Mask2Former's panoptic postprocessing,
        # it will filter mask area where score is less than 0.5 .
        filter_low_score=True),
    init_cfg=None)




data_root = ''
metainfo = dict(classes=('blood_vessel', ), palette=[(220, 20, 60)])
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_size = 1200

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),

    # dict(type='Mosaic', img_scale=(img_size, img_size), pad_val=114.0),

    # dict(
    #     type='MixUp',
    #     img_scale=(img_size, img_size),
    #     ratio_range=(0.8, 1.6),
    #     pad_val=114.0),
    # dict(type='RandomCrop', crop_size=(2048, 2048), cat_max_ratio=0.75),
    
    dict(type='Resize', img_scale=[(img_size, img_size)], keep_ratio=True),

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
        }], 
        #           [{
        #     'type': 'Translate',
        #     'prob': 0.4,
        #     'level': 5
        # }],
                # [{
                #       'type': 'PhotoMetricDistortion',
                #       'brightness_delta': 32,
                #       'contrast_range': (0.5, 1.5),
                #       'hue_delta': 18
                #   }],
                  # [{
                  #     'type': 'MinIoURandomCrop',
                  #     'min_ious': (0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                  #     'min_crop_size': 0.3
                  # }],
                  # [{
                  #     'type':
                  #     'CutOut',
                  #     'n_holes': (5, 10),
                  #     'cutout_shape': [(4, 4), (4, 8), (8, 4), (8, 8),
                  #                      (16, 32), (32, 16), (32, 32), (32, 48),
                  #                      (48, 32), (48, 48)]
                  # }],
                  
                  # [
                  #    {
                  #       'type': 'BrightnessTransform',
                  #             'prob': 0.6,
                  #             'level': 4
                  #         }, ],
                   
                   
                   
                  #  [{
                  #     'type': 'ColorTransform',
                  #     'prob': 1.0,
                  #     'level': 6
                  # }, 
   
                   
                   
                   [{
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
                p=0.4),
            # dict(type='RandomRotate90', p=0.4),
            # dict(type='RandomGamma',p=0.1),
            # dict(type='CLAHE',p=0.1)


        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes', gt_masks='masks'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(img_size, img_size)],
        flip=True,
        flip_direction=['horizontal', 'vertical'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    pin_memory=True,
    drop_last=False,
    train=dict(
        type='CocoDataset',
        classes=('blood_vessels', ),
        ann_file='/home/nischay/hubmap/coco/ds2wsiall_coco_1024_train_fold1.json',
        img_prefix='/home/nischay/hubmap/Data/train/',
        pipeline=train_pipeline
                   
       ),
    val=dict(
        type='CocoDataset',
        classes=('blood_vessels', ),
        ann_file='/home/nischay/hubmap/coco/ds1_coco_1024_valid_all_fold1.json',
        img_prefix='/home/nischay/hubmap/Data/train/',
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        classes=('blood_vessels', ),
        ann_file=
        '/home/nischay/hubmap/coco/ds12_coco_1024_valid_all_fold1.json',
        img_prefix='/home/nischay/hubmap/Data/train/',
        pipeline=test_pipeline
    
    
    ))


optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0001,)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.001,
    min_lr=0.02)
evaluation = dict(interval=1, metric=['segm'], save_best='segm_mAP')
runner = dict(type='EpochBasedRunner', max_epochs=8)
checkpoint_config = dict(interval=-1, filename_tmpl='detectors_epoch_{}.pth')
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
fp16 = None #dict(loss_scale=dict(init_scale=512))
gpu_ids = range(0, 3)
seed = 69
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None #'/home/nischay/hubmap/vitadap/detection/htc++_beit_adapter_large_fpn_3x_coco.pth.tar'
work_dir = './pret_dir/exp1_beitmask2former_1200_ds2wsiall'
workflow = [('train', 1)]
auto_resume = False
resume_from = None
launcher = 'none'

# optimizer_config = dict(grad_clip=None)
# checkpoint_config = dict(
#     interval=1,
#     max_keep_ckpts=2,
#     save_last=True,
# )
# fp16 = dict(loss_scale=dict(init_scale=512))
