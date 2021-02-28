from d2l import AllDeepLearning as d2l
from mxnet import gluon, image, nd
import os
import numpy as np

# Saved in the d2l package for later use
data_dir = "E:\\Python_Data"
voc_dir = "E:\\Python_Data\\VOCdevkit\\VOC2012"


def read_voc_images(voc_dir, is_train=True):
    text_fname = '%s/ImageSets/Segmentation/%s' % (
        voc_dir, 'train.txt' if is_train else 'val.txt')
    with open(text_fname, 'r') as f:
        images = f.read().split()
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in enumerate(images):
        features[i] = image.imread('%s/JPEGImages/%s.jpg' % (voc_dir, fname))
        labels[i] = image.imread('%s/SegmentationClass/%s.png' % (voc_dir, fname))
    return features, labels


train_features, train_labels = read_voc_images(voc_dir=voc_dir, is_train=True)
n = 5
# imgs = train_features[0:n] + train_labels[0:n]
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


# d2l.show_images(imgs, 2, n)
# d2l.plt.show()


def build_color_map2_label():
    color_map2_label = nd.zeros(256 ** 3)
    for i, color_map in enumerate(VOC_COLORMAP):
        color_map2_label[(color_map[0] * 256 + color_map[1]) * 256 + color_map[2]] = i
    return color_map2_label  # shape 256 ** 3 with value 0, 1,..VOC_COLORMAP.shape


def voc_label_indices(color_map, color_map2_label):
    color_map = color_map.astype(np.int32)  # color_map (train_labels with shape XxYxZ)
    idx = ((color_map[:, :, 0] * 256 + color_map[:, :, 1]) * 256
           + color_map[:, :, 2])  # sum color_map return shape (XxY)
    return color_map2_label[idx]


def voc_rand_crop(feature, label, height, width):
    """

    :param feature: feature img
    :param label: label img
    :param height: height img crop
    :param width: width img crop
    :return: feature, label img croped
    """
    feature, rect = image.random_crop(feature, (width, height))
    label = image.fixed_crop(label, *rect)
    return feature, label


# class VOCSegDataSet(gluon.data.Dataset):
#     def __init__(self, is_train, crop_size, voc_dir):
#         self.rgb_mean = nd.array([0.485, 0.456, 0.406])
#         self.rgb_std = nd.array([0.229, 0.224, 0.225])
#         self.crop_size = crop_size
#         features, labels = read_voc_images(voc_dir, is_train=is_train)
#         self.features = [self.normalize_image(feature) for feature in self.filter(features)]
#         self.labels = [self.filter(labels)]
#         self.color_map2_label = build_color_map2_label()
#         print('read ' + str(len(self.features)) + ' examples')
#
#     def normalize_image(self, img):
#         return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std
#
#     def filter(self, imgs):
#         return [img for img in imgs if (
#                 img.shape[0] >= self.crop_size[0] and
#                 img.shape[1] >= self.crop_size[1])]
#
#     def __getitem__(self, idx):
#         feature, label = voc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
#         return (feature.transpose((2, 0, 1),
#                                   voc_label_indices(label, self.color_map2_label)))
#
#     def __len__(self):
#         return len(self.features)


class VOCSegDataSet(gluon.data.Dataset):
    """A customized dataset to load VOC dataset."""

    def __init__(self, is_train, crop_size, voc_dir):
        self.rgb_mean = nd.array([0.485, 0.456, 0.406])
        self.rgb_std = nd.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = build_color_map2_label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, imgs):
        return [img for img in imgs if (
                img.shape[0] >= self.crop_size[0] and
                img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature.transpose((2, 0, 1)),
                voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

def load_data_voc(batch_size, crop_size):
    voc_dir = d2l.download_voc_pascal(data_dir)
    num_workers = d2l.get_dataloader_workers()
    train_iter = gluon.data.DataLoader(VOCSegDataSet(True, crop_size, voc_dir),
                                       batch_size, shuffle=True, last_batch='discard',
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(VOCSegDataSet(False, crop_size, voc_dir),
                                      batch_size, shuffle=True, last_batch='discard',
                                      num_workers=num_workers)
    return train_iter, test_iter


crop_size = (320, 480)
voc_train = d2l.VOCSegDataset(True, crop_size, voc_dir)
voc_test = d2l.VOCSegDataset(False, crop_size, voc_dir)

batch_size = 64
print(d2l.get_dataloader_workers())
train_iter = gluon.data.DataLoader(voc_train, batch_size, shuffle=True, last_batch='discard')

for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
# imgs = []
# for _ in range(n):
#     imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
# print("imgs", imgs)
# print("imgs indices", imgs[::2])
# d2l.show_images(imgs[::2] + imgs[1::2], 2, n)
# d2l.plt.show()


# y = voc_label_indices(train_labels[0], build_color_map2_label())
# print(y[105:115, 130:140], VOC_CLASSES[1])
