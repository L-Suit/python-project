_base_ = [
    '../dab_detr/dab-detr_r50_8xb2-50e_coco.py'
]


# dataset settings
dataset_type = 'CocoDataset'
#data_root = '/root/dataset/for31-weatherv2/images'
data_root = 'C:/ProgramData/lsh-dataset/for31-weatherv2/images'

backend_args = None

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=31)))

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# batch size, num workers
train_dataloader = dict(
    batch_size=16,
    num_workers=2)


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
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.0005))

auto_scale_lr = dict(enable=False, base_batch_size=16)

