import zipfile
from mxnet import gluon, nd
from d2l import AllDeepLearning as d2l


def read_date_nmt():
    file_name = gluon.utils.download('http://data.mxnet.io/data/fra-eng.zip')
    with zipfile.ZipFile(file_name, 'r') as f:
        return f.read('fra.txt').decode("utf-8")


raw_text = read_date_nmt()


def preprocess_nmt(text):
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')

    def no_space(char, prev_char):
        return (True if char in (',', '!', '.')
                and prev_char != ' ' else False)

    out = [' ' + char if i > 0 and no_space(char, text[i-1]) else char for i, char in enumerate(text.lower())]
    return ''.join(out)


text = preprocess_nmt(raw_text)
print(text[0:96])


def tokenize_mnt(text, num_examples=None):
    source, tager = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            tager.append(parts[1].split(' '))
    return source, tager


source, target = tokenize_mnt(text)
print(source[0:3], target[0:3])

# d2l.set_figsize((3.5, 2.5))
# d2l.plt.hist([[len(l) for l in source], [len(l) for l in target]], label=['source', 'target'])
# d2l.plt.legend(loc='upper right')
# d2l.plt.show()

src_vocab = d2l.Vocab(source, min_freq=3, use_special_tokens=True)


def trim_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


def build_array(lines, vocab, num_steps, is_source):
    lines = [vocab[l] for l in lines]
    if not is_source:
        lines = [[vocab.bos] + l + [vocab.eos] for l in lines]
    array = nd.array([trim_pad(l, num_steps, vocab.pad) for l in lines])
    valid_len = (array != vocab.pad).sum(axis=1)
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=1000):
    text = preprocess_nmt(read_date_nmt())
    source, target = tokenize_mnt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=3, use_special_tokens=True)
    tgt_vocab = d2l.Vocab(target, min_freq=3, use_special_tokens=True)
    src_array, src_valid_len = build_array(
        source, src_vocab, num_steps, True
    )
    tgt_array, tgt_valid_len = build_array(
        target, tgt_vocab, num_steps, False
    )
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return src_vocab, tgt_vocab, data_iter


src_vocab, tgt_vocab, train_iter = load_data_nmt(batch_size=2, num_steps=8)
for X, X_vlen, Y, Y_vlen, in train_iter:
    print('X = ', X.astype('int32'), '\nValid lengths for X=', X_vlen,
          '\bY = ', Y.astype('int32'), '\nValid lengths for Y=', Y_vlen)
    break


