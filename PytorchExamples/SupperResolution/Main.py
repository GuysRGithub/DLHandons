import argparse
from math import log10

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from AI.PytorchExamples.SupperResolution.Model import Net
from AI.PytorchExamples.SupperResolution.Data import get_training_set, get_test_set


parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=3, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=12, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, run without gpu")

torch.manual_seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")

print("===> Building model")
train_set = get_training_set(opt.upscale_factor)
test_set = get_test_set(opt.upscale_factor)

training_loader = DataLoader(dataset=train_set, num_workers=opt.threads,
                             batch_size=opt.batchSize, shuffle=True)
testing_loader = DataLoader(dataset=test_set, num_workers=opt.threads,
                            batch_size=opt.testBatchSize, shuffle=False)

print("===> Building model")
model = Net(opt.upscale_factor).to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)


def train(epoch):

    epoch_loss = 0
    for iteration, batch in enumerate(training_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}"
              .format(epoch, iteration, len(training_loader),
                      loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_loader)))


def test():
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_loader:
            inout, target = batch[0].to(device), batch[1].to(device)

            prediction = model(inout)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
        print("==> AVg PSNR: {:.4f} dB".format(avg_psnr / len(testing_loader)))


def checkpotin(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        test()
        checkpotin(epoch)

