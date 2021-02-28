from AI.AILearning.NaturalLanguageProcessing import NaturalLanguageStatistics as loader
from mxnet.gluon import rnn
from d2l import AllDeepLearning as d2l
batch_size, num_steps = 32, 36
train_iter, vocab = loader.load_data_time_machine(batch_size, num_steps)
gru_layer = rnn.GRU(256)
model = d2l.RNNModel(gru_layer, len(vocab))
d2l.train_ch8(model, train_iter, vocab, 1, 500, d2l.try_gpu())
d2l.plt.show()