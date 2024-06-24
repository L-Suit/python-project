import torch
import d2l.torch
from d2l import torch as d2l

torch.set_printoptions(2)  # 精简输出精度


img = d2l.plt.imread('../data/img/catdog.jpg')
h, w = img.shape[:2]
print(f'长宽：{h, w}')

X = torch.rand(size=(1, 3, h, w))
Y = d2l.multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape)

boxes = Y.reshape(h, w, 5, 4)

print(f'第250行250列像素为中心的锚框：{boxes[250, 250, 0, :]}')



d2l.set_figsize()
bbox_scale = torch.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
d2l.show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
                ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
d2l.plt.show()


ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.3, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
d2l.show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
d2l.show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
d2l.plt.show()


labels = d2l.multibox_target(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))
print(labels[2])
