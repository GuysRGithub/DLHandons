from d2l import AllDeepLearning as d2l
from mxnet import gluon, nd
import pandas as pd


def read_data_ml100k():
    data_dir = "E:/Python_Data/ml-100k/ml-100k/"
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(data_dir + 'u.data', '\t', names=names,
                       engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items


def split_data_ml100k(data, num_users, num_items, split_mode="random", test_ratio=0.1):
    """Split the dataset in random mode or seq-aware mode."""
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = [True if x == 1 else False for x in nd.random.uniform(0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data


def load_data_ml100k(data, num_users, num_items, feedback='explicit'):
    users, items, scores = [], [], []
    inter = nd.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter


def split_and_load_ml100k(split_mode='seq-aware', feedback="explict",
                          test_ratio=0.1, batch_size=256):
    data, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(data, num_users, num_items,
                                              split_mode, test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(test_data, num_users, num_items, feedback)
    train_set = gluon.data.ArrayDataset(nd.array(train_u), nd.array(train_i), nd.array(train_r))
    test_set = gluon.data.ArrayDataset(nd.array(test_u), nd.array(test_i), nd.array(test_r))
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True, last_batch='rollover')
    test_iter = gluon.data.DataLoader(test_set, batch_size=batch_size)
    return num_users, num_items, train_iter, test_iter






# data, num_users, num_items = read_data_ml100k()
# sparsity = 1 - len(data) / (num_users * num_items)
# d2l.plt.hist(data['rating'], bins=5, ec='black')
# d2l.plt.xlabel("Rating")
# d2l.plt.ylabel("Count")
# d2l.plt.title("Distribution of Ratings")
# d2l.plt.show()
