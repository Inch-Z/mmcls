_base_ = [
    '../_base_/models/segformer.py', '../_base_/datasets/imagenet_bs64_mixer_224.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='mit_b0',
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=256,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)),
    # model training and testing settings
    # train_cfg=dict(),
    # test_cfg=dict(mode='whole'))
)
data=dict(samples_per_gpu=256,workers_per_gpu=4)
evaluation = dict(interval=200, metric='accuracy')
resume_from='/workspace/mmclassification-master/work_dirs/fnet/latest.pth'
