from d2l import AllDeepLearning as d2l
from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import nn
from AI.AILearning.ComputerVision import ObjectDetectionDataset as loader


def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,
                     padding=1)


def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)


def forward(x, block):
    block.initialize()
    return block(x)


def flatten_pred(pred):
    return nd.flatten(pred.transpose((0, 2, 3, 1)))


def concat_preds(preds):
    return nd.concat(*[flatten_pred(p) for p in preds], dim=1)


def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_channels),
                nn.Activation('relu'))  # keep shape
    blk.add(nn.MaxPool2D(2))  # half shape
    return blk


def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk


def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalAvgPool2D()
    else:
        blk = down_sample_blk(128)
    return blk


def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = nd.contrib.MultiBoxPrior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return Y, anchors, cls_preds, bbox_preds


sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1


class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            setattr(self, 'blk_%d' % i, get_blk(i))
            setattr(self, 'cls_%d' % i, cls_predictor(num_anchors, num_classes))
            setattr(self, 'bbox_%d' % i, bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = \
                blk_forward(X, getattr(self, 'blk_%d' % i), sizes[i],
                            ratios[i], getattr(self, 'cls_%d' % i),
                            getattr(self, 'bbox_%d' % i))

        anchors = nd.concat(*anchors, dim=1)
        cls_preds = concat_preds(cls_preds)  # concat predict
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


bbox_loss = gluon.loss.L1Loss()


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox


def cls_eval(cls_preds, cls_labels):
    return float((cls_preds.argmax(axis=-1) == cls_labels).sum().asscalar())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((nd.abs((bbox_labels - bbox_preds) * bbox_masks)).sum().asscalar())


num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
batch_size = 16
train_iter, _ = loader.load_data_pikachu(batch_size)
ctx, net = d2l.try_gpu(), TinySSD(num_classes=1)
net.initialize(init=init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.2, 'wd': 5e-4})
cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()

for epoch in range(num_epochs):
    metric = d2l.Accumulator(4)
    train_iter.reset()
    for batch in train_iter:
        timer.start()
        X = batch.data[0].as_in_context(ctx)
        Y = batch.label[0].as_in_context(ctx)
        with autograd.record():
            # Generate multiscale anchor boxes and predict the category and
            # offset of each
            anchors, cls_preds, bbox_preds = net(X)
            # Label the category and offset of each anchor box
            bbox_labels, bbox_masks, cls_labels = \
                nd.contrib.MultiBoxTarget(anchors, Y, cls_preds.transpose((0, 2, 1)))
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)

        l.backward()
        trainer.step(batch_size)
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.size,
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks), bbox_labels.size)
    cls_err, bbox_mae = 1 - metric[0]/metric[1], metric[2]/metric[3]
    animator.add(epoch+1, (cls_err, bbox_mae))

img = image.imread('E:\\Python_Data\\pikachu.jpg')
feature = image.imresize(img, 256, 256).astype('float32')
X = nd.expand_dims(feature.transpose((2, 0, 1)), axis=0)


def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_context(ctx))
    cls_probs = nd.softmax(cls_preds).transpose((0, 2, 1))
    output = nd.contrib.MultiBoxDetection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]


out = predict(X)


def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img.asnumpy())
    for row in output:
        score = float(row[1].asscalar())
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')


display(img, out, threshold=0.3)
d2l.plt.show()
# net = TinySSD(1)
# net.initialize()
# X = nd.zeros((32, 3, 256, 256))
# anchors, cls_preds, bbox_preds = net(X)
#
# print('output anchors:', anchors.shape)
# print('output class preds:', cls_preds.shape)
# print('output bbox preds:', bbox_preds.shape)


# print(forward(nd.zeros((2, 3, 256, 256)), base_net()).shape)

# Y1 = forward(nd.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
# Y2 = forward(nd.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
# print(concat_preds([Y1, Y2]).shape)
