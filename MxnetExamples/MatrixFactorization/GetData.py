import os
import mxnet as mx


def get_movie_lens_data(prefix):
    if not os.path.exists("%s.zip" % prefix):
        print("Download dataset movie lens...")
        os.system("wget http://files.grouplens.org/datasets/movielens/%s.zip" % prefix)
        os.system("unzip %s.zip" % prefix)
        os.system("cd ml-10M100K; sh split_ratings.sh; cd -;")


def get_movie_lens_iter(filename, batch_size):
    """

    :param filename:
    :param batch_size:
    :return:
    """
    print("Preparing data iterators for " + filename + "...")
    user = []
    item = []
    score = []

    with open(filename, 'r') as f:
        num_samples = 0
        for line in f:
            tks = line.strip().split("::")
            if len(tks) != 4:
                continue
            num_samples += 1
            user.append(tks[0])
            item.append(tks[1])
            score.append(tks[2])
    user = mx.nd.array(user, dtype='int32')
    item = mx.nd.array(item)
    score = mx.nd.array(score)

    data_train = {'user': user, 'item': item}
    label_train = {'score': score}
    iter_train = mx.io.NDArrayIter(data=data_train, label=label_train,
                                   batch_size=batch_size, shuffle=True)
    return iter_train


