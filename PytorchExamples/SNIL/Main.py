import os
import time
import glob
import torch
import torch.optim as opt
from torchtext import data
from torchtext import datasets

from AI.PytorchExamples.SNIL.model import SNLIClassifier
from AI.PytorchExamples.SNIL.Utils import get_args, makedirs

args = get_args()
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda:{}'.format(args.gpu))
else:
    device = torch.device('cpu')

inputs = data.Field(lower=args.lower, tokenize='spacy')
answers = data.Field(sequential=False)

train, dev, test = datasets.SNLI.split(inputs, answers)

inputs.build_vocab(train, dev, test)
if args.word_vectors:
    if os.path.isfile(args.vector_cache):
        inputs.vocab.vectors = torch.load(args.vector_cache)
    else:
        inputs.vocab.load_vectors(args.word_vectors)
        makedirs(os.path.dirname(args.vector_cache))
        torch.save(inputs.vocab.vectors, args.vector_cache)
answers.build_vocab(train)

train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
                                                             batch_size=args.batch_size,
                                                             device=device)

config = args
config.n_embed = len(inputs.vocab)
config.n_cells = config.n_layers

if config.birnn:
    config.n_cells *= 2

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=device)
else:
    model = SNLIClassifier(config)
    if args.word_vectors:
        model.embed.weight.data.copy_(inputs.vocab.vectors)
        model.to(device)

criterion = torch.nn.CrossEntropyLoss()
op = opt.Adam(model.parameters(), lr=args.lr)

iterations = 0
start = time.time()
best_dev_acc = -1
header = ' Time Epoch Iteration Progress  (%Epoch) Loss Dev/Loss  Acc Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} '
                            '{:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
makedirs(args.save_path)
print(header)

for epoch in range(args.epochs):
    train_iter.init_epoch()
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(train_iter):

        model.train()
        op.zero_grad()

        iterations += 1

        answer = model(batch)

        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
        n_total += batch.batch_size
        train_acc = 100.0 * n_correct / n_total

        loss = criterion(answer, batch.label)
        if iterations % args.save_every == 0:
            snapshot_prefix = os.path.join(args.save_path, 'snapshot')
            snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{.6f}_iter_{}' \
                                              '_model.pt'.format(train_acc,
                                                                 loss,
                                                                 iterations)
            torch.save(model, snapshot_path)
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

        if iterations % args.dev_every == 0:

            model.eval()
            dev_iter.init_epoch()

            n_dev_correct, dev_loss = 0, 0
            with torch.no_grad():
                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                    answer = model(dev_batch)
                    n_dev_correct += (
                                torch.max(answer, 1)[1].view(dev_batch.label.size()) == dev_batch.label).sum().item()
                    dev_loss = criterion(answer, dev_batch.label)
            dev_acc = 100.0 * n_dev_correct / len(dev)

            print(dev_log_template.format(time.time() - start),
                  100.0 * (1 + batch_idx) / len(train_iter), loss.item(),
                  dev_loss.item(),
                  train_acc, dev_acc)

            if dev_loss > best_dev_acc:
                best_dev_acc = dev_acc
                snapshot_prefix = os.path.join(args.save_path, "best_snapshot")
                snapshot_path = snapshot_prefix + '_dev_acc_{}_dev_loss_{}__iter_{}_model.pt'.format(dev_acc,
                                                                                                     dev_loss.item(),
                                                                                                     iterations)
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + "*"):
                    if f != snapshot_path:
                        os.remove(f)

            elif iterations % args.log_every == 0:
                print(log_template.format(time.time() - start,
                                          epoch, iterations, 1 + batch_idx, len(train_iter),
                                          100. * (1 + batch_idx) / len(train_iter), loss.item(), ' ' * 8,
                                          n_correct / n_total * 100, ' ' * 12))

