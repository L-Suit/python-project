_base_ = [
    '../centernet/centernet_r18-dcnv2_8xb16-crop512-140e_coco.py'
]


# dataset settings
dataset_type = 'CocoDataset'
data_root = '/root/dataset/for31-weatherv2/images'

backend_args = None

model = dict(
    bbox_head=dict(num_classes=31))

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# batch size, num workers
train_dataloader = dict(
    batch_size=32,
    num_workers=8)


# learning rate
param_scheduler = [
    dict(type='CosineAnnealingLR',
         T_max=200,
         by_epoch=True,
         begin=0,
         end=200)
]

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0005))

auto_scale_lr = dict(enable=False, base_batch_size=16)

