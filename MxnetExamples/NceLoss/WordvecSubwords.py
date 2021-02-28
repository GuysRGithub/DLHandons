import logging
from optparse import OptionParser
import mxnet as mx
from AI.MxnetExamples.NceLoss.NCE import NceAuc
from AI.MxnetExamples.NceLoss.Text8Data import DataIterSubWords
from AI.MxnetExamples.NceLoss.WordvecNet import get_subword_net

head = '%s(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format=head)

EMBEDDING_SIZE = 100
BATCH_SIZE = 256
NUM_LABEL = 5
NUM_EPOCH = 20
MIN_COUNT = 5  # only works when doing nagative sampling, keep it same as nce-loss
GRAMS = 3  # here we use triple-letter representation
MAX_SUBWORDS = 10
PADDING_CHAR = '</s>'

if __name__ == '__main__':
    head = '%s(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    parser = OptionParser()
    parser.add_option("-g", "--gpu", action="store_true",
                      dest="gpu", default=False, help="use gpu")
    options, args = parser.parse_args()

    batch_size = BATCH_SIZE
    num_label = NUM_LABEL
    embedding_size = EMBEDDING_SIZE

    data_train = DataIterSubWords("./data/text8",
                                  batch_size=batch_size,
                                  num_label=num_label,
                                  min_count=MIN_COUNT,
                                  gram=GRAMS,
                                  max_subwords=MAX_SUBWORDS,
                                  padding_char=PADDING_CHAR)
    print(data_train.vocab_size)
    network = get_subword_net(data_train.vocab_size, num_input=num_label - 1,
                              embedding_size=embedding_size)
    ctx = mx.cpu()
    if options.gpu:
        ctx = mx.gpu()

    model = mx.mod.Module(
        symbol=network,
        data_names=[x[0] for x in data_train.provide_data],
        label_names=[x[0] for x in data_train.provide_label],
        context=[ctx])

    print("Training on {}".format("GPU" if options.gpu else "CPU"))
    metric = NceAuc()
    model.fit(train_data=data_train,
              num_epoch=NUM_EPOCH,
              optimizer='sgd',
              optimizer_params={'learning_rate': 0.3, 'momentum': 0.9, 'wd': 0.0000},
              initializer=mx.initializer.Xavier(factor_type='in', magnitude=2.34),
              eval_metric=metric,
              batch_end_callback=mx.callback.Speedometer(batch_size, 50))
