#!/usr/bin/env python3

"""
Bonito training.
"""

import os
import csv
import sys
from datetime import datetime
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter

from bonito.model import Model
from bonito.util import load_data, init
from bonito.training import ChunkDataSet, train, test
from bonito.TextColor import TextColor

import toml
import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

try: from apex import amp
except ImportError: pass


def handle_output_directory(output_dir):
    """
    Process the output directory and return a valid directory where we save the output
    :param output_dir: Output directory path
    :return:
    """
    timestr = time.strftime("%m%d%Y_%H%M%S")
    # process the output directory
    if output_dir[-1] != "/":
        output_dir += "/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create an internal directory so we don't overwrite previous runs
    model_save_dir = output_dir + "trained_models_" + timestr + "/"
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    stats_directory = model_save_dir + "stats_" + timestr + "/"

    if not os.path.exists(stats_directory):
        os.mkdir(stats_directory)

    return model_save_dir, stats_directory


def save_model(model, optimizer, epoch, file_name):
    """
    Save the best model
    :param model: A trained model
    :param optimizer: Model optimizer
    :param epoch: Epoch/iteration number
    :param file_name: Output file name
    :return:
    """
    if os.path.isfile(file_name):
        os.remove(file_name)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epochs': epoch,
    }, file_name)
    sys.stderr.write(TextColor.RED + "MODEL SAVED SUCCESSFULLY.\n" + TextColor.END)
    sys.stderr.flush()


def main(args):
    # create the output directory
    model_directory, stats_directory = handle_output_directory(os.path.abspath(args.output_directory))

    # initialize the training
    gpu_mode = args.gpu_mode
    init(args.seed, gpu_mode)

    # load training data
    input_directory = os.path.abspath(args.input_directory)

    sys.stderr.write(TextColor.GREEN + "INFO: LOADING " + str(args.chunk_size) +
                     " CHUNKS FROM: " + str(input_directory) + "\n" + TextColor.END)
    sys.stderr.flush()

    chunks, chunk_lengths, targets, target_lengths = load_data(input_directory, limit=args.chunk_size, shuffle=True)
    sys.stderr.write(TextColor.GREEN + "INFO: LOADED " + str(len(chunks)) +
                     " CHUNKS SUCCESSFULLY" + "\n" + TextColor.END )
    sys.stderr.flush()

    # split training data into train-test
    split = np.floor(chunks.shape[0] * args.validation_split).astype(np.int32)
    sys.stderr.write(TextColor.GREEN + "INFO: SPLITTING TRAIN DATA. TRAIN: " + str(int(args.validation_split * 100))
                     + "% TEST: " + str(int(100-args.validation_split * 100)) + "%\n" + TextColor.END)
    sys.stderr.flush()

    train_dataset = ChunkDataSet(chunks[:split], chunk_lengths[:split], targets[:split], target_lengths[:split])
    test_dataset = ChunkDataSet(chunks[split:], chunk_lengths[split:], targets[split:], target_lengths[split:])
    sys.stderr.write(TextColor.PURPLE + "INFO: TOTAL TRAIN CHUNKS:\t" + str(len(train_dataset)) + "\n" + TextColor.END)
    sys.stderr.write(TextColor.PURPLE + "INFO: TOTAL TEST CHUNKS:\t" + str(len(test_dataset)) + "\n" + TextColor.END)
    sys.stderr.flush()

    if not train_dataset:
        sys.stderr.write(TextColor.RED + "ERROR: NO CHUNKS AVAILABLE FOR TRAINING\n" + TextColor.END)
        sys.stderr.flush()
        exit(1)
    if not test_dataset:
        sys.stderr.write(TextColor.RED + "ERROR: NO CHUNKS AVAILABLE FOR TEST\n" + TextColor.END)
        sys.stderr.flush()
        exit(1)

    # train and test dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # load the model
    config = toml.load(args.config)
    argsdict = dict(training=vars(args))

    sys.stderr.write(TextColor.GREEN + "INFO: LOADING MODEL\n" + TextColor.END)
    sys.stderr.flush()
    model = Model(config)
    if gpu_mode:
        model = model.cuda()

    model.train()
    sys.stderr.write(TextColor.GREEN + "INFO: MODEL LOADED\n" + TextColor.END)

    # this is for re-training but needs to be updated, this is not a proper way to load model weights
    # optimizer weights need to be loaded too.
    # weights = os.path.join(model_directory, 'weights.tar')
    #
    # # if path exists, then load weights?
    # if os.path.exists(weights):
    #     model.load_state_dict(torch.load(weights))

    # save the config file to the model directory
    toml.dump({**config, **argsdict}, open(os.path.join(model_directory, 'config.toml'), 'w'))
    optimizer = AdamW(model.parameters(), amsgrad=True, lr=args.learning_rate)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sys.stderr.write(TextColor.CYAN + "INFO: TOTAL TRAINABLE PARAMETERS:\t" + str(param_count) + "\n" + TextColor.END)

    if args.nvidia_apex:
        try:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
        except NameError:
            sys.stderr.write(TextColor.RED + "ERROR : Cannot use AMP: Apex package needs to be installed manually, "
                                             "See https://github.com/NVIDIA/apex\n")
            sys.stderr.flush()
            exit(1)

    # get the stride and alphabets. They will not be a part of the model object if we wrap it around dataparallel class
    stride = config['block'][0]['stride'][0]
    alphabet = config['labels']['labels']

    if gpu_mode:
        model = torch.nn.DataParallel(model).cuda()

    scheduler = CosineAnnealingLR(optimizer, args.epochs * len(train_loader), eta_min=0, last_epoch=-1)
    log_interval = np.floor(len(train_dataset) / args.batch_size * 0.10)

    for epoch in range(1, args.epochs + 1):

        train_loss, duration = train(
            log_interval, model, gpu_mode, train_loader, optimizer, epoch, stride, alphabet, use_amp=args.nvidia_apex
        )
        test_loss, mean, median = test(model, gpu_mode, test_loader, stride, alphabet)

        model_filename = "BONITO_MODEL_EPOCH_" + str(epoch) + ".pkl"
        save_model(model, optimizer, epoch, os.path.join(model_directory, model_filename))

        # perform the test
        with open(os.path.join(stats_directory, 'training.csv'), 'a', newline='') as csvfile:
            csvw = csv.writer(csvfile, delimiter=',')
            if epoch == 1:
                csvw.writerow([
                    'time', 'duration', 'epoch', 'train_loss',
                    'validation_loss', 'validation_mean', 'validation_median'
                ])
            csvw.writerow([
                datetime.today(), int(duration), epoch,
                train_loss, test_loss, mean, median,
            ])

        scheduler.step(epoch)


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument(
        "-i",
        "--input_directory",
        type=str,
        required=True,
        help="Path to the input directory."
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        type=str,
        required=True,
        help="Path to the output directory."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Config file for the model."
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        required=False,
        default=1e-3,
        help="Learning rate."
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        required=False,
        default=25,
        help="Random seed."
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        required=False,
        default=1,
        help="Epoch size for training iteration."
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        required=False,
        default=16,
        help="Batch size."
    )
    parser.add_argument(
        "-cs",
        "--chunk_size",
        type=int,
        required=False,
        default=1000000,
        help="Number of chunks to train on."
    )
    parser.add_argument(
        "-vs",
        "--validation_split",
        type=float,
        required=False,
        default=0.99,
        help="Fraction of data to be used for training. Rest will be used for testing"
    )
    parser.add_argument(
        "-amp",
        "--nvidia_apex",
        default=False,
        action='store_true',
        help="Use nvidia apex for mixed precision and distributed training."
    )
    parser.add_argument(
        "-g",
        "--gpu_mode",
        default=False,
        action='store_true',
        help="If set then PyTorch will use GPUs for inference. CUDA required."
    )
    parser.add_argument(
        "-f",
        "--force",
        default=False,
        action='store_true',
        help="Not sure exactly what it does."
    )
    return parser
