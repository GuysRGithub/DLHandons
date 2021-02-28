from collections import defaultdict
from d2l import AllDeepLearning as d2l
from mxnet import gluon, nd
import numpy as np

data_dir = "E:/Python_Data/ctr/ctr/"


def custom_cumsum(array):
    new_array = list()
    for i, value in enumerate(array):
        if i == 0:
            new_array.append(array[i].asscalar().astype('float32'))
        else:
            new_array.append((new_array[i - 1] + array[i]).asscalar().astype('float32'))
    return new_array


class CTRDataset(gluon.data.Dataset):
    """
    Fix bug by create custom cumsum method
    """
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = nd.zeros(self.NUM_FEATS, dtype=np.int64)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                label = np.float32([0, 0])
                label[int(values[0])] = 1
                instance['y'] = [np.float32(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if
                               c >= min_threshold} for i, cnt in feat_cnts.items()}
            self.feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)}
                                for i, cnt in feat_mapper.items()}
            self.defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = np.array((0, *(custom_cumsum(self.field_dims)[:-1])))

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        feat = np.array([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                         for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, self.data[idx]['y']


train_data = CTRDataset(data_path=data_dir + "train.csv")
# print(train_data[0])
# train_iter = gluon.data.DataLoader(
#     train_data, shuffle=True, last_batch="rollover", batch_size=2048,
#     num_workers=0)
# for values in enumerate(train_iter):
#     print(values)
#     break

# test = nd.array([1, 2, 3, 4, 5, 6])
# print(cumsum(test))
"""
####################3                      SUMMARY      ################################
    Click-through rate is an important metric that is used to measure the 
effectiveness of advertising systems and recommender systems.

    Click-through rate prediction is usually converted to a binary 
    classification problem. The target is to predict whether an ad/item will be
     clicked or not based on given features.
"""