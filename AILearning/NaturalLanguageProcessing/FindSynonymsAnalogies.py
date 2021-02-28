from mxnet import nd
from mxnet.contrib import text

glove_6b50d = text.embedding.create('glove', pretrained_file_name='glove.6B.50d.txt')


# print(len(glove_6b50d))
# We can use a word to get its index in the dictionary, or we can get the word from its index.

# print(glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367])


def knn(W, x, k):
    # The added 1e-9 is for numerical stability
    cos = nd.dot(W, x.reshape(-1)) / (
            nd.sqrt(nd.sum(W * W, axis=1) + 1e-9) * nd.sqrt((x * x).sum()))
    topk = nd.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i.asscalar())] for i in topk]


def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed.get_vecs_by_tokens([query_token]), k + 1)

    for i, c in zip(topk[1:], cos[1:]):
        print('cosine similar=%.3f: %s' % (c.asscalar(), (embed.idx_to_token[int(i.asscalar())])))


def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed.get_vecs_by_tokens([token_a, token_b, token_c])
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0].asscalar())]  # Remove unknown words


print(get_analogy('bad', 'worst', 'tall', glove_6b50d))