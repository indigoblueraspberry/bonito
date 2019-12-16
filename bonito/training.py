"""
Bonito train
"""

import time
import sys
from itertools import starmap
from tqdm import tqdm

from bonito.util import decode_ctc, decode_ref, accuracy
from bonito.TextColor import TextColor

import torch
import numpy as np
import torch.nn as nn

try: from apex import amp
except ImportError: pass


criterion = nn.CTCLoss(reduction='mean')


class ChunkDataSet:
    def __init__(self, chunks, chunk_lengths, targets, target_lengths):
        self.chunks = np.expand_dims(chunks, axis=1)
        self.chunk_lengths = chunk_lengths
        self.targets = targets
        self.target_lengths = target_lengths

    def __getitem__(self, i):
        return (
            self.chunks[i],
            self.chunk_lengths[i].astype(np.int32),
            self.targets[i].astype(np.int32),
            self.target_lengths[i].astype(np.int32)
        )

    def __len__(self):
        return len(self.chunks)


def train(log_interval, model, gpu_mode, train_loader, optimizer, epoch, stride, alphabet, use_amp=False):

    t0 = time.perf_counter()
    chunks = 0
    loss = 0
    model.train()
    sys.stderr.write(TextColor.BLUE + "INFO: TRAINING STARTING ON EPOCH: " + str(epoch) + "\n")
    progress_bar = tqdm(total=len(train_loader), desc='Train loss', leave=True, ncols=100)
    for batch_idx, (data, out_lengths, target, lengths) in enumerate(train_loader, start=1):

        optimizer.zero_grad()

        chunks += data.shape[0]

        if gpu_mode:
            data = data.cuda()
            target = target.cuda()

        log_probs = model(data)

        loss = criterion(log_probs.transpose(0, 1), target, out_lengths / stride, lengths)

        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        progress_bar.refresh()
        progress_bar.update(1)
        progress_bar.set_description("Loss: " + str(loss.item()))
    progress_bar.close()

    sys.stderr.write(TextColor.GREEN + "INFO: TRAIN LOSS: " + str(loss.item()) + "\n")

    return loss.item(), time.perf_counter() - t0


def test(model, gpu_mode, test_loader, stride, alphabet):

    model.eval()
    test_loss = None
    predictions = []
    prediction_lengths = []

    sys.stderr.write(TextColor.YELLOW + "INFO: TEST STARTING" + "\n")
    with torch.no_grad():
        progress_bar = tqdm(total=len(test_loader), desc='Test loss', leave=True, ncols=100)
        for batch_idx, (data, out_lengths, target, lengths) in enumerate(test_loader, start=1):
            if gpu_mode:
                data, target = data.cuda(), target.cuda()

            log_probs = model(data)
            if test_loss is None:
                test_loss = criterion(log_probs.transpose(1, 0), target, out_lengths / stride, lengths)
            else:
                test_loss += criterion(log_probs.transpose(1, 0), target, out_lengths / stride, lengths)

            predictions.append(torch.exp(log_probs).cpu())
            prediction_lengths.append(out_lengths / stride)

            progress_bar.refresh()
            progress_bar.update(1)
            progress_bar.set_description("Test loss: " + str(test_loss.item()))
    progress_bar.close()

    predictions = np.concatenate(predictions)
    lengths = np.concatenate(prediction_lengths)

    references = [decode_ref(target, alphabet) for target in test_loader.dataset.targets]
    sequences = [decode_ctc(post[:n], alphabet) for post, n in zip(predictions, lengths)]

    if all(map(len, sequences)):
        accuracies = list(starmap(accuracy, zip(references, sequences)))
    else:
        accuracies = [0]

    mean = np.mean(accuracies)
    median = np.median(accuracies)
    sys.stderr.write(TextColor.GREEN + "Validation Loss:              %.4f" % (test_loss / batch_idx) + "\n")
    sys.stderr.write("Validation Accuracy (mean):   %.3f%%" % max(0, mean) + "\n")
    sys.stderr.write("Validation Accuracy (median): %.3f%%" % max(0, median) + "\n" + TextColor.END)

    return test_loss.item() / batch_idx, mean, median
