from d2l import AllDeepLearning as d2l
from mxnet import contrib, image, nd

img = image.imread('E:\\Python_Data\\dog_cat.jpg')
h, w = img.shape[0:2]


def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize((3.5, 2.5))
    fmap = nd.zeros((1, 10, fmap_w, fmap_h))
    anchors = nd.contrib.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = nd.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)


display_anchors(fmap_w=2, fmap_h=2, s=[0.8])
d2l.plt.show()
