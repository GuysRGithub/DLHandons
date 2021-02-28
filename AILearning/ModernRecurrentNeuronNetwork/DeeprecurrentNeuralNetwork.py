from d2l import AllDeepLearning as d2l
from mxnet.gluon import rnn
from AI.AILearning.NaturalLanguageProcessing import NaturalLanguageStatistics as loader

batch_size, num_steps = 32, 35
train_iter, vocab = loader.load_data_time_machine(batch_size, num_steps)
vocab_size, num_hiddens, num_layers, ctx = len(vocab), 256, 2, d2l.try_gpu()
lstm_layer = rnn.LSTM(num_hiddens, 2)
model = d2l.RNNModel(lstm_layer, len(vocab))
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, ctx)
d2l.plt.show()
