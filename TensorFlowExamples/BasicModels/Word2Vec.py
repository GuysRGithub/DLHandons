import collections
import random
import zipfile
import numpy as np
import tensorflow as tf

learning_rate = .1
batch_size = 128
num_steps = 3000000
display_step = 10000
eval_step = 200000

eval_words = [b'five', b'of', b'going', b'hardware', b'american', b'britain']

embedding_size = 200  # Dimension of the embedding vector.
max_vocabulary_size = 50000  # Total number of different words in the vocabulary.
min_occurrence = 10  # Remove all words that does not appears at least n times.
skip_window = 3  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.

url = "http://mattmahoney.net/dc/text8.zip"
data_path = "E:/Python_Data/text8.zip"
data_dir = "E:/Python_Data/text8/text8"
with zipfile.ZipFile(data_path) as f:
    line = f.read(f.namelist()[0])
    text_words = line.split()

# Build the dictionary and replace rare words with UNK token.
count = [('UNK', -1)]
# Retrieve the most common words.
count.extend(collections.Counter(text_words).most_common(max_vocabulary_size))


for i in range(len(count) - 1, -1, -1):
    if count[i][1] < min_occurrence:
        count.pop(i)
    else:
        # The collection is ordered, so stop when 'min_occurrence' is reached.
        break

# Compute the vocabulary size.
vocabulary_size = len(count)
word2id = dict()

# Assign an id to each word.
for i, (word, _) in enumerate(count):
    word2id[word] = i

data = list()
unk_count = 0
for word in text_words:
    # Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary.
    index = word2id.get(word, 0)
    if index == 0:
        unk_count += 1
    data.append(index)
count[0] = ('UNK', unk_count)
id2word = dict(zip(word2id.values(), word2id.keys()))

data_idx = 0


def next_batch(batch_size, num_skips, skips_window):
    global data_idx
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=batch_size, dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # get window size (words left and right + current one).
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    if data_idx + span > len(data):
        data_idx = 0
    buffer.extend(data[data_idx:data_idx + span])
    data_idx += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + 1] = buffer[skip_window]
            labels[i * num_skips + 1, 0] = buffer[context_word]
        if data_idx == len(data):
            buffer.extend(data[0:span])
            data_idx = span
        else:
            buffer.append(data[data_idx])
            data_idx += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch.
    data_idx = (data_idx + len(data) - span) % len(data)
    return batch, labels


with tf.device('/cpu:0'):
    embedding = tf.Variable(tf.random.normal((vocabulary_size, embedding_size)))
    nce_weights = tf.Variable(tf.random.normal((vocabulary_size, embedding_size)))
    nce_biases = tf.Variable(tf.zeros(shape=vocabulary_size))


def get_embedding(x):
    with tf.device('/cpu:0'):
        # Lookup the corresponding embedding vectors for each sample in X.
        x_embed = tf.nn.embedding_lookup(embedding, x)
        return x_embed


def nce_loss(x_embed, y):
    with tf.device('/cpu:0'):
        # Compute the average NCE loss for the batch.
        y = tf.cast(y, tf.int64)
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                             biases=nce_biases,
                                             labels=y,
                                             inputs=x_embed,
                                             num_sampled=num_sampled,
                                             num_classes=vocabulary_size))
    return loss


def evaluate(x_embed):
    with tf.device('/cpu:0'):
        # Compute the cosine similarity between input data embedding and every embedding vectors
        x_embed = tf.cast(x_embed, tf.float32)
        x_embed_norm = x_embed / tf.sqrt(tf.reduce_sum(tf.square(x_embed)))
        embedding_norm = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding), 1,
                                                           keepdims=True), tf.float32)
        cosine_sim_op = tf.matmul(x_embed_norm, embedding_norm, transpose_b=True)
        return cosine_sim_op


def run_optimization(x, y):
    with tf.device('/cpu:0'):
        # Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g:
            emb = get_embedding(x)
            loss = nce_loss(emb, y)

        # Compute gradients.
        gradients = g.gradient(loss, [embedding, nce_weights, nce_biases])

        # Update W and b following gradients.
        optimizer.apply_gradients(zip(gradients, [embedding, nce_weights, nce_biases]))


optimizer = tf.optimizers.SGD(learning_rate)

# Words for testing.
x_test = np.array([word2id[w] for w in eval_words])

# Run training for the given number of steps.
for step in range(1, num_steps + 1):
    batch_x, batch_y = next_batch(batch_size, num_skips, skip_window)
    run_optimization(batch_x, batch_y)

    if step % display_step == 0 or step == 1:
        loss = nce_loss(get_embedding(batch_x), batch_y)
        print("step: %i, loss: %f" % (step, loss))

    if step % display_step == 0 or step == 1:
        print("Evaluation...")
        sim = evaluate(get_embedding(x_test)).numpy()
        for i in range(len(eval_words)):
            top_k = 8  # number of nearest neighbors.
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = '"%s" nearest neighbors' % eval_words
            for k in range(top_k):
                log_str = '%s, %s' % (log_str, id2word[nearest[k]])
            print(log_str)
