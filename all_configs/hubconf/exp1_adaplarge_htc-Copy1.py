drop_path_rate = 0.4
model = dict(
    type='HybridTaskCascadeAug',
    backbone=dict(
        type='ViTAdapter',
        img_size=384,
        pretrain_size=384,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        drop_path_rate=drop_path_rate,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=True,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        window_attn=[True, True, True, True, True, False,
                     True, True, True, True, True, False,
                     True, True, True, True, True, False,
                     True, True, True, True, True, False],
        window_size=[14, 14, 14, 14, 14, None,
                     14, 14, 14, 14, 14, None,
                     14, 14, 14, 14, 14, None,
                     14, 14, 14, 14, 14, None],
        pretrained=None),
    neck=[
        dict(
            type='ExtraAttention',
            in_channels=[1024, 1024, 1024, 1024],
            num_head=32,
            with_ffn=True,
            ffn_ratio=4.0,
            drop_path=drop_path_rate,
        ),
        dict(
            type='FPN',
            in_channels=[1024, 1024, 1024, 1024],
            out_channels=256,
            num_outs=5)],
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
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
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
                type='Shared4Conv1FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared4Conv1FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared4Conv1FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
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
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
        ],
),
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
        ]),
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
            mask_thr_binary=0.5),
        aug=dict(
            score_thr=0.001,
            nms=dict(type='soft_nms', iou_threshold=0.5),
            max_per_img=1000,
            scale_ranges=[['l'], ['l']],
        )
    ))
# optimizer
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# augmentation strategy originates from DETR / Sparse RCNN
# file_client_args = dict(backend='petrel')
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    # dict(type='InstaBoost',
         # action_candidate=('normal', 'horizontal', 'skip'),
         # action_prob=(1, 0, 0),
         # scale=(0.8, 1.2),
         # dx=15,
         # dy=15,
         # theta=(-1, 1),
         # color_prob=0.5,
         # hflag=False,
         # aug_ratio=0.5),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=False),
    dict(type='Resize',
         img_scale=[(512, 512)],
         # multiscale_mode='range',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(512, 512)],
        flip=True,
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
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=6,
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
        '/home/nischay/hubmap/coco/ds12_coco_1024_valid_all_fold1.json',
        img_prefix='/home/nischay/hubmap/Data/train/',
        pipeline=test_pipeline))

# optimizer = dict(
                 # type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 # constructor='LayerDecayOptimizerConstructor',
                 # paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.90))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001,)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.001,
    min_lr=1e-08)
evaluation = dict(interval=1, metric=['segm'], save_best='segm_mAP')
runner = dict(type='EpochBasedRunner', max_epochs=18)
checkpoint_config = dict(interval=-1, filename_tmpl='detectors_epoch_{}.pth')
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
fp16 = None #dict(loss_scale=dict(init_scale=512))
gpu_ids = range(0, 3)
seed = 69
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/nischay/hubmap/vitadap/detection/htc++_augreg_adapter_large_fpn_3x_coco.pth'
work_dir = './work_dirs/exp1_adaplargehtc'
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
