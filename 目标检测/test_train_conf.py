_base_ = [
    'configs/_base_/models/faster-rcnn_r50_fpn.py',
    'configs/_base_/datasets/coco_detection.py',
    'configs/_base_/default_runtime.py'
]
# 模型配置
# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=102)))


train_cfg = dict(
    type='EpochBasedTrainLoop',  # 训练循环的类型，请参考 https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=12,  # 最大训练轮次
    val_interval=1)  # 验证间隔。每个 epoch 验证一次
val_cfg = dict(type='ValLoop')  # 验证循环的类型
test_cfg = dict(type='TestLoop')  # 测试循环的类型

optim_wrapper = dict(  # 优化器封装的配置
    type='OptimWrapper',  # 优化器封装的类型。可以切换至 AmpOptimWrapper 来启用混合精度训练
    optimizer=dict(  # 优化器配置。支持 PyTorch 的各种优化器。请参考 https://pytorch.org/docs/stable/optim.html#algorithms
        type='SGD',  # 随机梯度下降优化器
        lr=0.02,  # 基础学习率
        momentum=0.9,  # 带动量的随机梯度下降
        weight_decay=0.0001),  # 权重衰减
    clip_grad=None,  # 梯度裁剪的配置，设置为 None 关闭梯度裁剪。使用方法请见 https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html
    )
# 自动调整学习率 关闭
auto_scale_lr = dict(enable=False, base_batch_size=16)

# 数据集
dataset_type = 'CocoDataset'  # 数据集类型，这将被用来定义数据集。
data_root = 'D:\dataset\ip102\Detection/'  # 数据的根路径。
metainfo = {
    'classes': ('rice leaf roller',
                'rice leaf caterpillar',
                'paddy stem maggot',
                'asiatic rice borer',
                'yellow rice borer',
                'rice gall midge',
                'Rice Stemfly',
                'brown plant hopper',
                'white backed plant hopper',
                'small brown plant hopper',
                'rice water weevil',
                'rice leafhopper',
                'grain spreader thrips',
                'rice shell pest',
                'grub',
                'mole cricket',
                'wireworm',
                'white margined moth',
                'black cutworm',
                'large cutworm',
                'yellow cutworm',
                'red spider',
                'corn borer',
                'army worm',
                'aphids',
                'Potosiabre vitarsis',
                'peach borer',
                'english grain aphid',
                'green bug',
                'bird cherry-oataphid',
                'wheat blossom midge',
                'penthaleus major',
                'longlegged spider mite',
                'wheat phloeothrips',
                'wheat sawfly',
                'cerodonta denticornis',
                'beet fly',
                'flea beetle',
                'cabbage army worm',
                'beet army worm',
                'Beet spot flies',
                'meadow moth',
                'beet weevil',
                'sericaorient alismots chulsky',
                'alfalfa weevil',
                'flax budworm',
                'alfalfa plant bug',
                'tarnished plant bug',
                'Locustoidea',
                'lytta polita',
                'legume blister beetle',
                'blister beetle',
                'therioaphis maculata Buckton',
                'odontothrips loti',
                'Thrips',
                'alfalfa seed chalcid',
                'Pieris canidia',
                'Apolygus lucorum',
                'Limacodidae',
                'Viteus vitifoliae',
                'Colomerus vitis',
                'Brevipoalpus lewisi McGregor',
                'oides decempunctata',
                'Polyphagotars onemus latus',
                'Pseudococcuscomstocki Kuwana',
                'parathrene regalis',
                'Ampelophaga',
                'Lycorma delicatula',
                'Xylotrechus',
                'Cicadella viridis',
                'Miridae',
                'Trialeurodes vaporariorum',
                'Erythroneura apicalis',
                'Papilio xuthus',
                'Panonchus citri McGregor',
                'Phyllocoptes oleiverus ashmead',
                'Icerya purchasi Maskell',
                'Unaspis yanonensis',
                'Ceroplastes rubens',
                'Chrysomphalus aonidum',
                'Parlatoria zizyphus Lucus',
                'Nipaecoccus vastalor',
                'Aleurocanthus spiniferus',
                'Tetradacus c Bactrocera minax',
                'Dacus dorsalis(Hendel)',
                'Bactrocera tsuneonis',
                'Prodenia litura',
                'Adristyrannus',
                'Phyllocnistis citrella Stainton',
                'Toxoptera citricidus',
                'Toxoptera aurantii',
                'Aphis citricola Vander Goot',
                'Scirtothrips dorsalis Hood',
                'Dasineura sp',
                'Lawana imitata Melichar',
                'Salurnis marginella Guerr',
                'Deporaus marginatus Pascoe',
                'Chlumetia transversa',
                'Mango flat beak leafhopper',
                'Rhytidodera bowrinii white',
                'Sternochetus frigidus',
                'Cicadellidae',
                )
    # 'palette': [
    #     (220, 20, 60),
    # ]
}


train_dataloader = dict(  # 训练 dataloader 配置
    batch_size=2,  # 单个 GPU 的 batch size
    num_workers=2,  # 单个 GPU 分配的数据加载线程数
    persistent_workers=True,  # 如果设置为 True，dataloader 在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    sampler=dict(  # 训练数据的采样器
        type='DefaultSampler',  # 默认的采样器，同时支持分布式和非分布式训练。请参考 https://mmengine.readthedocs.io/zh_CN/latest/api/generated/mmengine.dataset.DefaultSampler.html#mmengine.dataset.DefaultSampler
        shuffle=True),  # 随机打乱每个轮次训练数据的顺序
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # 批数据采样器，用于确保每一批次内的数据拥有相似的长宽比，可用于节省显存
    dataset=dict(  # 训练数据集的配置
        type=dataset_type,
        data_root=data_root,
        ann_file='coco/voc07_trainval.json',  # 标注文件路径
        data_prefix=dict(img=''),  # 图片路径前缀
        filter_cfg=dict(filter_empty_gt=True, min_size=32)  # 图片和标注的过滤配置
        ))  # 这是由之前创建的 train_pipeline 定义的数据处理流程。
val_dataloader = dict(  # 验证 dataloader 配置
    batch_size=1,  # 单个 GPU 的 Batch size。如果 batch-szie > 1，组成 batch 时的额外填充会影响模型推理精度
    num_workers=2,  # 单个 GPU 分配的数据加载线程数
    persistent_workers=True,  # 如果设置为 True，dataloader 在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    drop_last=False,  # 是否丢弃最后未能组成一个批次的数据
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),  # 验证和测试时不打乱数据顺序
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='coco/voc07_test.json',
        data_prefix=dict(img=''),
        test_mode=True  # 开启测试模式，避免数据集过滤图片和标注
        ))
test_dataloader = val_dataloader  # 测试 dataloader 配置

# 评测器
val_evaluator = dict(  # 验证过程使用的评测器
    type='CocoMetric',  # 用于评估检测和实例分割的 AR、AP 和 mAP 的 coco 评价指标
    ann_file=data_root + 'coco/voc07_test.json',  # 标注文件路径
    metric=['bbox'],  # 需要计算的评价指标，`bbox` 用于检测，`segm` 用于实例分割
    format_only=False)
test_evaluator = val_evaluator  # 测试过程使用的评测器
