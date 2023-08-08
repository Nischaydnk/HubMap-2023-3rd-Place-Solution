
model = dict(
    type='HybridTaskCascade',
    # pretrained='',
    backbone=dict(
        type='ResNeSt',
        stem_channels=128,
        depth=200,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='open-mmlab://resnest200')
    ),
    neck=dict(
        type='FPN_CARAFE',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        upsample_cfg=dict(
            type='carafe',
            up_kernel=5,
            up_group=1,
            encoder_kernel=3,
            encoder_dilation=1,
            compressed_channels=64
        )
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)
    ),
    roi_head=dict(
        type='HybridTaskCascadeRoIHead',
        interleaved=True,
        mask_info_flow=True,
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=21,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=21,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=21,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=[
            dict(
                type='HTCMaskHead',
                with_conv_res=False,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=21,
                upsample_cfg=dict(
                    type='carafe',
                    scale_factor=2,
                    up_kernel=5,
                    up_group=1,
                    encoder_kernel=3,
                    encoder_dilation=1,
                    compressed_channels=64),
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=21,
                upsample_cfg=dict(
                    type='carafe',
                    scale_factor=2,
                    up_kernel=5,
                    up_group=1,
                    encoder_kernel=3,
                    encoder_dilation=1,
                    compressed_channels=64),
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=21,
                upsample_cfg=dict(
                    type='carafe',
                    scale_factor=2,
                    up_kernel=5,
                    up_group=1,
                    encoder_kernel=3,
                    encoder_dilation=1,
                    compressed_channels=64),
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
        ],
    ),
)

# model training and testing settings
train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ])
test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='soft_nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5))



data_root = ''
metainfo = dict(classes=('blood_vessel', ), palette=[(220, 20, 60)])
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.15,
        rotate_limit=15,
        p=0.4),
    dict(type='RandomRotate90', p=0.4)
]
img_size = 2048
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
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
        }], [{
            'type': 'Translate',
            'prob': 0.4,
            'level': 5
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
                                       (16, 32), (32, 16), (32, 32), (32, 48),
                                       (48, 32), (48, 48)]
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
                  }]
                 
                 
                 ]),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.15,
                rotate_limit=15,
                p=0.4),
            # dict(type='RandomRotate90', p=0.4)
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
        ann_file='/home/nischay/hubmap/coco/ds1_coco_1024_train_all_fold1.json',
        img_prefix='/home/nischay/hubmap/Data/train/',
        pipeline=train_pipeline),
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
        '/home/nischay/hubmap/coco/ds1_coco_1024_valid_all_fold1.json',
        img_prefix='/home/nischay/hubmap/Data/train/',
        pipeline=test_pipeline))
# optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.00001,)
optimizer = dict(
                 type='AdamW', lr=0.00007, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.80))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=150,
    warmup_ratio=0.001,
    min_lr=1e-8)
evaluation = dict(interval=1, metric=['segm'], save_best='segm_mAP')
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=-1, filename_tmpl='detectors_epoch_{}.pth')
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
fp16 = dict(loss_scale=dict(init_scale=512))
gpu_ids = range(0, 3)
seed = 69
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
# load_from = '/home/nischay/hubmap/try_mm/cbnet_pret/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth'
load_from = '/home/nischay/hubmap/vitadap/detection/dir_noleak/res200d/exp1_newtype_1600_nodrop_f1/best_segm_mAP_epoch_34.pth'
# load_from = '/home/nischay/hubmap2/mask_cascade_rcnn_ResNeSt_200_FPN_dcn_syncBN_all_tricks_3x-e1901134.pth'
work_dir = './dir_noleak_onlyds12/exp1_res200d/pretexp1_newtype_1600_nodrop_f1_2048'
workflow = [('train', 1)]
auto_resume = False
resume_from = None
launcher = 'none'
device = 'cuda'
