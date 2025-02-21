# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    # dict(
    #     # 具体参数意义见 https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.optim.OneCycleLR.html#mmengine.optim.OneCycleLR
    #     type='OneCycleLR',
    #     eta_max=0.001,
    #     total_steps=200,
    #     pct_start = 0.015,    #200轮中多少比例是先增加学习率的
    #     by_epoch=True,
    #     div_factor =10,
    #     final_div_factor=10,
    #     verbose=True
    # )
# 在 [0, 100) 迭代时使用线性学习率
#     dict(type='LinearLR',
#          start_factor=0.001,
#          by_epoch=False,
#          begin=0,
#          end=3,
#          verbose=True),
    # 在 [100, 900) 迭代时使用余弦学习率
    dict(type='CosineAnnealingLR',
         T_max=200,
         by_epoch=True,
         begin=0,
         end=200,
         verbose=True)
]

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.0005))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
