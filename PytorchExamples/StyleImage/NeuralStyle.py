import argparse
import os
import sys
import time
import re
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.onnx

from AI.PytorchExamples.StyleImage import Utils
from AI.PytorchExamples.StyleImage.TransformerNet import TransformNet
from AI.PytorchExamples.StyleImage.VGG import Vgg16


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    transformer = TransformNet().to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)

    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    style = Utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    features_style = vgg(Utils.normalize_batch(style))
    gram_style = [Utils.gram_matrix(y) for y in features_style]

    for e in range(args.epochs):
        print("Training epoch", e)
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = .0
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)

            y = transformer(x)

            y = Utils.normalize_batch(y)
            x = Utils.normalize_batch(x)

            feature_y = vgg(y)
            feature_x = vgg(x)

            content_loss = args.content_weight * mse_loss(feature_y.relu2,
                                                          feature_x.relu2)
            style_loss = 0.
            for ft_y, gm_s in zip(feature_y, gram_style):
                gm_y = Utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % \
                    args.checkpoint_interval == 0:
                transformer.eval().cpu()
                ck_point_model_file_name = "ck_point_epoch_" + str(e) + \
                                           "_batch_id_" + str(batch_id + 1) + ".pth"
                ck_point_model_path = os.path.join(args.checkpoint_model_dir,
                                                   ck_point_model_file_name)
                torch.save(transformer.state_dict(), ck_point_model_path)
                transformer.to(device).train()
    transformer.eval().cpu()
    save_model_file_name = "epoch_" + str(args.epochs) + "_" + \
                           str(args.style_weight) + ".model"

    # str(time.ctime()).replace(' ', '_') + \
                           # "_" + str(args.content_weight) + "_" + \
    save_model_path = os.path.join(args.save_model_dir, save_model_file_name)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model save at %s", save_model_path)


def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = Utils.load_image(args.content_image,
                                     scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    if args.model.endswith(".onnx"):
        pass
        output = stylize_onnx_caffe2(content_image, args)
    else:
        with torch.no_grad():
            style_model = TransformNet()
            state_dict = torch.load(args.model)

            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            if args.export_onnx:
                assert args.export_onnx.endswith(".onnx"), "Export model file" \
                                                           "should end with .onnx"
                output = torch.onnx._export(style_model, content_image,
                                            args.export_onnx).cpu()
            else:
                output = style_model(content_image).cpu()
    Utils.save_image(args.output_image, output[0])


def stylize_onnx_caffe2(content_image, args):

    assert not args.export_onnx

    import onnx
    import caffe2.python.onnx.backend as backend

    model = onnx.load(args.model)

    prepared_backed = backend.prepare(model, device="CUDA")
    inp = {model.graph.input[0].name: content_image.numpy()}
    c2_out = prepared_backed.run(inp)[0]

    return torch.from_numpy(c2_out)


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")

    main_arg_parser.add_argument("--cuda", type=int, default=1,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    subparsers = main_arg_parser.add_subparsers(title="subcommand", dest="subcommand",)

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")

    train_arg_parser.add_argument("--epochs", type=int, default=10000,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=2,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, default="saved_models",
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, default=0,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, default="images/content-images/amber.jpg",
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, default="images/output-images/rain.jpg",
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, default="saved_models/test.model",
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch "
                                      "path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, default=0,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--export_onnx", type=str,
                                 help="export ONNX model to a given file")

    args = main_arg_parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)

    else:
        args = eval_arg_parser.parse_args()
        stylize(args)


if __name__ == '__main__':
    main()


