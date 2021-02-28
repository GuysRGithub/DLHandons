from d2l import AllDeepLearning as d2l
from mxnet import gluon, init, nd, image
from AI.AILearning.ComputerVision import SemanticSegmentation as loader
from mxnet.gluon import nn

data_dir = "E:\\Python_Data"

pre_trained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
print(pre_trained_net.features[-4:], pre_trained_net.output)


net = nn.HybridSequential()
for layer in pre_trained_net.features[:-2]:
    net.add(layer)
num_classes = 21
# # X = nd.random.normal(shape=(1, 3, 320, 480))
net.add(nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(num_classes, kernel_size=64, padding=16, strides=32))
# net.initialize()


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (nd.arange(kernel_size).reshape(-1, 1),
          nd.arange(kernel_size).reshape(1, -1))
    filt = (1 - nd.abs(og[0] - center) / factor) * \
           (1 - nd.abs(og[1] - center) / factor)
    weight = nd.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[:in_channels, :out_channels, :, :] = filt
    return nd.array(weight)


def predict(img):
    X = test_iter._dataset.normalize_image(img)
    X = nd.expand_dims(X.transpose((2, 0, 1)), axis=0)
    pred = net(X.as_in_context(ctx[0])).argmax(axis=1)
    return pred.reshape(pred.shape[1], pred.shape[2])


def label2image(pred):
    colormap = nd.array(d2l.VOC_COLORMAP, ctx=ctx[0], dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]


conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize(init.Constant(bilinear_kernel(3, 3, 4)))

img = image.imread('E:/Python_Data/dog_cat.jpg')
X = nd.expand_dims(img.astype('float32').transpose((2, 0, 1)), axis=0) / 255
Y = conv_trans(X)
out_img = Y[0].transpose((1, 2, 0))
# d2l.set_figsize((3.5, 2.5))
# print('input image shape:', img.shape)
# # d2l.plt.imshow(img.asnumpy())
# print('output image shape:', out_img.shape)
# d2l.plt.imshow(out_img.asnumpy())
# d2l.plt.show()

W = bilinear_kernel(num_classes, num_classes, 64)
net[-1].initialize(init.Constant(W))
net[-2].initialize(init=init.Xavier())
batch_size, crop_size = 8, (320, 480)
train_iter, test_iter = loader.load_data_voc(batch_size, crop_size)

num_epochs, lr, wd, ctx = 5, 0.1, 1e-3, d2l.try_all_gpus()
loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'wd': wd})
voc_dir = d2l.download_voc_pascal(data_dir)
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 480, 320)
    X = image.fixed_crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X, pred, image.fixed_crop(test_labels[i], *crop_rect)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
d2l.train_ch12(net, train_iter, test_iter, loss, trainer, num_epochs, ctx)
d2l.plt.show()