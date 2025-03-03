# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

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


model = dict(
    backbone=dict(
        deepen_factor=0.33,
        widen_factor=0.5),
    neck=dict(in_channels=[128, 256, 512], out_channels=128, num_csp_blocks=1),
    bbox_head=dict(in_channels=128, feat_channels=128, exp_on_reg=False,num_classes=31))
