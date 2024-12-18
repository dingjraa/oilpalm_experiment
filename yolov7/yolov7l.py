_backend_args = None
_multiscale_resize_transforms = [
    dict(
        transforms=[
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                320,
                320,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    320,
                    320,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                960,
                960,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    960,
                    960,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
]
anchors = [
    [
        (
            127,
            181,
        ),
        (
            192,
            121,
        ),
        (
            191,
            192,
        ),
    ],
    [
        (
            145,
            256,
        ),
        (
            252,
            155,
        ),
        (
            188,
            254,
        ),
    ],
    [
        (
            256,
            211,
        ),
        (
            225,
            256,
        ),
        (
            257,
            257,
        ),
    ],
]
backend_args = None
base_lr = 0.01
batch_shapes_cfg = dict(
    batch_size=1,
    extra_pad_ratio=0.5,
    img_size=640,
    size_divisor=32,
    type='BatchShapePolicy')
class_name = ('Oil-Palm', )
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
]
data_root = './data/aa.v7i.coco-mmdetection/'
dataset_type = 'YOLOv5CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1000,
        save_best='auto',
        save_param_scheduler=False,
        type='CheckpointHook'),
    logger=dict(interval=1000, type='LoggerHook'),
    param_scheduler=dict(
        lr_factor=0.1,
        max_epochs=300,
        scheduler_type='cosine',
        type='YOLOv5ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scale = (
    640,
    640,
)
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
launcher = 'none'
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco/yolov7_l_syncbn_fast_8x16b-300e_coco_20221123_023601-8113c0eb.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
loss_bbox_weight = 0.05
loss_cls_weight = 0.3
loss_obj_weight = 0.7
lr_factor = 0.1
max_epochs = 300
max_keep_ckpts = 3
max_translate_ratio = 0.2
metainfo = dict(
    classes=('Oil-Palm', ), palette=[
        (
            220,
            20,
            60,
        ),
    ])
mixup_alpha = 8.0
mixup_beta = 8.0
mixup_prob = 0.15
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='L',
        frozen_stages=4,
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        type='YOLOv7Backbone'),
    bbox_head=dict(
        head_module=dict(
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=[
                256,
                512,
                1024,
            ],
            num_base_priors=3,
            num_classes=1,
            type='YOLOv7HeadModule'),
        loss_bbox=dict(
            bbox_format='xywh',
            iou_mode='ciou',
            loss_weight=0.05,
            reduction='mean',
            return_iou=True,
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=0.3,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_obj=dict(
            loss_weight=0.7,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        obj_level_weights=[
            4.0,
            1.0,
            0.4,
        ],
        prior_generator=dict(
            base_sizes=[
                [
                    (
                        12,
                        16,
                    ),
                    (
                        19,
                        36,
                    ),
                    (
                        40,
                        28,
                    ),
                ],
                [
                    (
                        36,
                        75,
                    ),
                    (
                        76,
                        55,
                    ),
                    (
                        72,
                        146,
                    ),
                ],
                [
                    (
                        142,
                        110,
                    ),
                    (
                        192,
                        243,
                    ),
                    (
                        459,
                        401,
                    ),
                ],
            ],
            strides=[
                8,
                16,
                32,
            ],
            type='mmdet.YOLOAnchorGenerator'),
        prior_match_thr=4.0,
        simota_candidate_topk=10,
        simota_cls_weight=1.0,
        simota_iou_weight=3.0,
        type='YOLOv7Head'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='YOLOv5DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        block_cfg=dict(
            block_ratio=0.25,
            middle_ratio=0.5,
            num_blocks=4,
            num_convs_in_block=1,
            type='ELANBlock'),
        in_channels=[
            512,
            1024,
            1024,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        out_channels=[
            128,
            256,
            512,
        ],
        type='YOLOv7PAFPN',
        upsample_feats_cat_first=False),
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.65, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    type='YOLODetector')
model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(iou_threshold=0.65, type='nms'),
    nms_pre=30000,
    score_thr=0.001)
mosiac4_pipeline = [
    dict(
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='Mosaic'),
    dict(
        border=(
            -320,
            -320,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=0.2,
        scaling_ratio_range=(
            0.1,
            2.0,
        ),
        type='YOLOv5RandomAffine'),
]
mosiac9_pipeline = [
    dict(
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='Mosaic9'),
    dict(
        border=(
            -320,
            -320,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=0.2,
        scaling_ratio_range=(
            0.1,
            2.0,
        ),
        type='YOLOv5RandomAffine'),
]
norm_cfg = dict(eps=0.001, momentum=0.03, type='BN')
num_classes = 1
num_det_layers = 3
num_epoch_stage2 = 30
obj_level_weights = [
    4.0,
    1.0,
    0.4,
]
optim_wrapper = dict(
    constructor='YOLOv7OptimWrapperConstructor',
    optimizer=dict(
        batch_size_per_gpu=16,
        lr=0.01,
        momentum=0.937,
        nesterov=True,
        type='SGD',
        weight_decay=0.0005),
    type='OptimWrapper')
param_scheduler = None
persistent_workers = True
pre_transform = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]
prior_match_thr = 4.0
randchoice_mosaic_pipeline = dict(
    prob=[
        0.8,
        0.2,
    ],
    transforms=[
        [
            dict(
                img_scale=(
                    640,
                    640,
                ),
                pad_val=114.0,
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                ],
                type='Mosaic'),
            dict(
                border=(
                    -320,
                    -320,
                ),
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                max_translate_ratio=0.2,
                scaling_ratio_range=(
                    0.1,
                    2.0,
                ),
                type='YOLOv5RandomAffine'),
        ],
        [
            dict(
                img_scale=(
                    640,
                    640,
                ),
                pad_val=114.0,
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                ],
                type='Mosaic9'),
            dict(
                border=(
                    -320,
                    -320,
                ),
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                max_translate_ratio=0.2,
                scaling_ratio_range=(
                    0.1,
                    2.0,
                ),
                type='YOLOv5RandomAffine'),
        ],
    ],
    type='RandomChoice')
randchoice_mosaic_prob = [
    0.8,
    0.2,
]
resume = False
save_epoch_intervals = 1
scaling_ratio_range = (
    0.1,
    2.0,
)
simota_candidate_topk = 10
simota_cls_weight = 1.0
simota_iou_weight = 3.0
strides = [
    8,
    16,
    32,
]
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=3,
    dataset=dict(
        ann_file='valid/_annotations.coco.json',
        batch_shapes_cfg=dict(
            batch_size=1,
            extra_pad_ratio=0.5,
            img_size=640,
            size_divisor=32,
            type='BatchShapePolicy'),
        data_prefix=dict(img='valid/'),
        data_root='./data/aa.v7i.coco-mmdetection/',
        metainfo=dict(classes=('Oil-Palm', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='./data/aa.v7i.coco-mmdetection//test/_annotations.coco.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(scale=(
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(
            640,
            640,
        ),
        type='LetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
        ),
        type='mmdet.PackDetInputs'),
]
train_ann_file = 'annotations/instances_train2017.json'
train_batch_size_per_gpu = 3
train_cfg = dict(
    dynamic_intervals=[
        (
            270,
            1,
        ),
    ],
    max_epochs=300,
    type='EpochBasedTrainLoop',
    val_interval=1)
train_data_prefix = 'train2017/'
train_dataloader = dict(
    batch_size=3,
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
        data_root='./data/aa.v7i.coco-mmdetection/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(classes=('Oil-Palm', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                prob=[
                    0.8,
                    0.2,
                ],
                transforms=[
                    [
                        dict(
                            img_scale=(
                                640,
                                640,
                            ),
                            pad_val=114.0,
                            pre_transform=[
                                dict(
                                    backend_args=None,
                                    type='LoadImageFromFile'),
                                dict(type='LoadAnnotations', with_bbox=True),
                            ],
                            type='Mosaic'),
                        dict(
                            border=(
                                -320,
                                -320,
                            ),
                            border_val=(
                                114,
                                114,
                                114,
                            ),
                            max_rotate_degree=0.0,
                            max_shear_degree=0.0,
                            max_translate_ratio=0.2,
                            scaling_ratio_range=(
                                0.1,
                                2.0,
                            ),
                            type='YOLOv5RandomAffine'),
                    ],
                    [
                        dict(
                            img_scale=(
                                640,
                                640,
                            ),
                            pad_val=114.0,
                            pre_transform=[
                                dict(
                                    backend_args=None,
                                    type='LoadImageFromFile'),
                                dict(type='LoadAnnotations', with_bbox=True),
                            ],
                            type='Mosaic9'),
                        dict(
                            border=(
                                -320,
                                -320,
                            ),
                            border_val=(
                                114,
                                114,
                                114,
                            ),
                            max_rotate_degree=0.0,
                            max_shear_degree=0.0,
                            max_translate_ratio=0.2,
                            scaling_ratio_range=(
                                0.1,
                                2.0,
                            ),
                            type='YOLOv5RandomAffine'),
                    ],
                ],
                type='RandomChoice'),
            dict(
                alpha=8.0,
                beta=8.0,
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        prob=[
                            0.8,
                            0.2,
                        ],
                        transforms=[
                            [
                                dict(
                                    img_scale=(
                                        640,
                                        640,
                                    ),
                                    pad_val=114.0,
                                    pre_transform=[
                                        dict(
                                            backend_args=None,
                                            type='LoadImageFromFile'),
                                        dict(
                                            type='LoadAnnotations',
                                            with_bbox=True),
                                    ],
                                    type='Mosaic'),
                                dict(
                                    border=(
                                        -320,
                                        -320,
                                    ),
                                    border_val=(
                                        114,
                                        114,
                                        114,
                                    ),
                                    max_rotate_degree=0.0,
                                    max_shear_degree=0.0,
                                    max_translate_ratio=0.2,
                                    scaling_ratio_range=(
                                        0.1,
                                        2.0,
                                    ),
                                    type='YOLOv5RandomAffine'),
                            ],
                            [
                                dict(
                                    img_scale=(
                                        640,
                                        640,
                                    ),
                                    pad_val=114.0,
                                    pre_transform=[
                                        dict(
                                            backend_args=None,
                                            type='LoadImageFromFile'),
                                        dict(
                                            type='LoadAnnotations',
                                            with_bbox=True),
                                    ],
                                    type='Mosaic9'),
                                dict(
                                    border=(
                                        -320,
                                        -320,
                                    ),
                                    border_val=(
                                        114,
                                        114,
                                        114,
                                    ),
                                    max_rotate_degree=0.0,
                                    max_shear_degree=0.0,
                                    max_translate_ratio=0.2,
                                    scaling_ratio_range=(
                                        0.1,
                                        2.0,
                                    ),
                                    type='YOLOv5RandomAffine'),
                            ],
                        ],
                        type='RandomChoice'),
                ],
                prob=0.15,
                type='YOLOv5MixUp'),
            dict(type='YOLOv5HSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='YOLOv5CocoDataset'),
    num_workers=1,
    persistent_workers=True,
    pin_memory=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_num_workers = 4
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        prob=[
            0.8,
            0.2,
        ],
        transforms=[
            [
                dict(
                    img_scale=(
                        640,
                        640,
                    ),
                    pad_val=114.0,
                    pre_transform=[
                        dict(backend_args=None, type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                    ],
                    type='Mosaic'),
                dict(
                    border=(
                        -320,
                        -320,
                    ),
                    border_val=(
                        114,
                        114,
                        114,
                    ),
                    max_rotate_degree=0.0,
                    max_shear_degree=0.0,
                    max_translate_ratio=0.2,
                    scaling_ratio_range=(
                        0.1,
                        2.0,
                    ),
                    type='YOLOv5RandomAffine'),
            ],
            [
                dict(
                    img_scale=(
                        640,
                        640,
                    ),
                    pad_val=114.0,
                    pre_transform=[
                        dict(backend_args=None, type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                    ],
                    type='Mosaic9'),
                dict(
                    border=(
                        -320,
                        -320,
                    ),
                    border_val=(
                        114,
                        114,
                        114,
                    ),
                    max_rotate_degree=0.0,
                    max_shear_degree=0.0,
                    max_translate_ratio=0.2,
                    scaling_ratio_range=(
                        0.1,
                        2.0,
                    ),
                    type='YOLOv5RandomAffine'),
            ],
        ],
        type='RandomChoice'),
    dict(
        alpha=8.0,
        beta=8.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                prob=[
                    0.8,
                    0.2,
                ],
                transforms=[
                    [
                        dict(
                            img_scale=(
                                640,
                                640,
                            ),
                            pad_val=114.0,
                            pre_transform=[
                                dict(
                                    backend_args=None,
                                    type='LoadImageFromFile'),
                                dict(type='LoadAnnotations', with_bbox=True),
                            ],
                            type='Mosaic'),
                        dict(
                            border=(
                                -320,
                                -320,
                            ),
                            border_val=(
                                114,
                                114,
                                114,
                            ),
                            max_rotate_degree=0.0,
                            max_shear_degree=0.0,
                            max_translate_ratio=0.2,
                            scaling_ratio_range=(
                                0.1,
                                2.0,
                            ),
                            type='YOLOv5RandomAffine'),
                    ],
                    [
                        dict(
                            img_scale=(
                                640,
                                640,
                            ),
                            pad_val=114.0,
                            pre_transform=[
                                dict(
                                    backend_args=None,
                                    type='LoadImageFromFile'),
                                dict(type='LoadAnnotations', with_bbox=True),
                            ],
                            type='Mosaic9'),
                        dict(
                            border=(
                                -320,
                                -320,
                            ),
                            border_val=(
                                114,
                                114,
                                114,
                            ),
                            max_rotate_degree=0.0,
                            max_shear_degree=0.0,
                            max_translate_ratio=0.2,
                            scaling_ratio_range=(
                                0.1,
                                2.0,
                            ),
                            type='YOLOv5RandomAffine'),
                    ],
                ],
                type='RandomChoice'),
        ],
        prob=0.15,
        type='YOLOv5MixUp'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
tta_model = dict(
    tta_cfg=dict(max_per_img=300, nms=dict(iou_threshold=0.65, type='nms')),
    type='mmdet.DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    transforms=[
                        dict(scale=(
                            640,
                            640,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                640,
                                640,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            320,
                            320,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                320,
                                320,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            960,
                            960,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                960,
                                960,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
            ],
            [
                dict(prob=1.0, type='mmdet.RandomFlip'),
                dict(prob=0.0, type='mmdet.RandomFlip'),
            ],
            [
                dict(type='mmdet.LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'pad_param',
                        'flip',
                        'flip_direction',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_ann_file = 'annotations/instances_val2017.json'
val_batch_size_per_gpu = 3
val_cfg = dict(type='ValLoop')
val_data_prefix = 'val2017/'
val_dataloader = dict(
    batch_size=3,
    dataset=dict(
        ann_file='valid/_annotations.coco.json',
        batch_shapes_cfg=dict(
            batch_size=1,
            extra_pad_ratio=0.5,
            img_size=640,
            size_divisor=32,
            type='BatchShapePolicy'),
        data_prefix=dict(img='valid/'),
        data_root='./data/aa.v7i.coco-mmdetection/',
        metainfo=dict(classes=('Oil-Palm', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='./data/aa.v7i.coco-mmdetection//valid/_annotations.coco.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
val_interval_stage2 = 1
val_num_workers = 4
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
weight_decay = 0.0005
work_dir = './work_dirs\\yolov7'
