from d2l import AllDeepLearning as d2l
from mxnet import autograd, gluon, nd
from mxnet.gluon import nn
from AI.AILearning.NaturalLanguageProcessing import WordEmbedding as loader

"""
    train data may include frequency, index,...

   The layer in which the obtained word is embedded is called the embedding layer, which can be obtained by creating an 
   nn.Embedding
   
   Embedding:  When we enter the index  i  of a word, the embedding layer returns the  ith  row of the weight matrix 
   as its word vector. 

"""

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = loader.load_data_ptb(batch_size, max_window_size, num_noise_words)

embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()


# X = nd.ones((2, 1, 4))
# Y = nd.ones((2, 4, 6))


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = nd.batch_dot(v, u.swapaxes(1, 2))
    return pred


loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
pred = nd.array([[.5] * 4] * 2)
label = nd.array([[1, 0, 1, 0]] * 2)
mask = nd.array([[1, 1, 1, 1], [1, 1, 0, 0]])
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size))
net.initialize()


def train(net, data_iter, lr, num_epochs, ctx=d2l.try_gpu()):
    net.initialize(ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[0, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)
        for i, batch in enumerate(data_iter):
            center, comtext_negative, mask, label = \
                [data.as_in_context(ctx) for data in batch]
            with autograd.record():
                pred = skip_gram(center, comtext_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask) /
                     mask.sum(axis=1) * mask.shape[1])
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum().asscalar(), l.size)
            if (i + 1) % 50 == 0:
                animator.add(epoch + (i + 1) / len(data_iter),
                             (metric[0] / metric[1],))
    print('loss %.3f, %d tokens/sec on %s ' % (
        metric[0] / metric[1], metric[1] / timer.stop(), ctx))


def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()  # vector weight
    x = W[vocab[query_token]]  # index
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = nd.dot(W, x) / nd.sqrt(nd.sum(W * W, axis=1) * nd.sum(x * x) + 1e-5)
    topk = nd.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:
        print('cosine sim=%.3f: %s' % (cos[i].asscalar(), (vocab.idx_to_token[i])))


lr, num_epochs = 0.01, 5
train(net, data_iter, lr, num_epochs)
get_similar_tokens('chip', 3, net[0])

# d2l.plt.show()
