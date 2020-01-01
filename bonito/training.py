"""
Bonito train
"""

import time
import sys
from itertools import starmap
from tqdm import tqdm

from bonito.util import decode_ctc_rle, decode_ref, accuracy
from bonito.TextColor import TextColor

import torch
import numpy as np
import torch.nn as nn

try: from apex import amp
except ImportError: pass


criterion_base = nn.CTCLoss(reduction='mean')
criterion_rle = nn.CTCLoss(reduction='mean')


class ChunkDataSet:
    def __init__(self, chunks, chunk_lengths, targets, target_lengths, rle_reference_bases, rles_reference_rles, rle_reference_lengths):
        self.chunks = np.expand_dims(chunks, axis=1)
        self.chunk_lengths = chunk_lengths
        self.targets = targets
        self.target_lengths = target_lengths
        self.rle_reference_bases = rle_reference_bases
        self.rles_reference_rles = rles_reference_rles
        self.rle_reference_lengths = rle_reference_lengths

    def __getitem__(self, i):
        return (
            self.chunks[i],
            self.chunk_lengths[i].astype(np.int32),
            self.targets[i].astype(np.int32),
            self.target_lengths[i].astype(np.int32),
            self.rle_reference_bases[i].astype(np.int32),
            self.rles_reference_rles[i].astype(np.int32),
            self.rle_reference_lengths[i].astype(np.int32)
        )

    def __len__(self):
        return len(self.chunks)


def train(log_interval, model, gpu_mode, train_loader, optimizer, epoch, stride, alphabet, use_amp=False):

    t0 = time.perf_counter()
    chunks = 0
    total_loss = 0
    model.train()
    sys.stderr.write(TextColor.BLUE + "INFO: TRAINING STARTING ON EPOCH: " + str(epoch) + "\n")
    progress_bar = tqdm(total=len(train_loader), desc='Train loss', leave=True, ncols=100)
    for batch_idx, (data, out_lengths, target, lengths, rle_bases, rle_rles, rle_lengths) in enumerate(train_loader, start=1):

        optimizer.zero_grad()

        chunks += data.shape[0]

        if gpu_mode:
            data = data.cuda()
            target = target.cuda()
            rle_bases = rle_bases.cuda()
            rle_rles = rle_rles.cuda()

        log_probs_base, log_probs_rle = model(data)

        base_loss = criterion_base(log_probs_base.transpose(0, 1), rle_bases, out_lengths / stride, rle_lengths)
        rle_loss = criterion_rle(log_probs_rle.transpose(0, 1), rle_rles, out_lengths / stride, rle_lengths)
        total_loss = base_loss + rle_loss

        if use_amp:
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()

        optimizer.step()
        progress_bar.refresh()
        progress_bar.update(1)
        progress_bar.set_description("BASE LOSS: " + str(base_loss.item()) + " RLE LOSS: " + str(rle_loss.item()))
    progress_bar.close()

    sys.stderr.write(TextColor.GREEN + "INFO: TOTAL LOSS: " + str(total_loss.item()) + "\n")

    return total_loss.item(), time.perf_counter() - t0


def test(model, gpu_mode, test_loader, stride, alphabet):

    model.eval()
    test_loss = None
    base_loss = None
    rle_loss = None
    predictions_bases = []
    predictions_rles = []
    prediction_lengths = []

    sys.stderr.write(TextColor.YELLOW + "INFO: TEST STARTING" + "\n")
    with torch.no_grad():
        progress_bar = tqdm(total=len(test_loader), desc='Test loss', leave=True, ncols=100)
        for batch_idx, (data, out_lengths, target, lengths, rle_bases, rle_rles, rle_lengths) in enumerate(test_loader, start=1):
            if gpu_mode:
                data = data.cuda()
                target = target.cuda()
                rle_bases = rle_bases.cuda()
                rle_rles = rle_rles.cuda()

            log_probs_base, log_probs_rle = model(data)
            if test_loss is None:
                base_loss = criterion_base(log_probs_base.transpose(0, 1), rle_bases, out_lengths / stride, rle_lengths)
                rle_loss = criterion_rle(log_probs_rle.transpose(0, 1), rle_rles, out_lengths / stride, rle_lengths)
                test_loss = base_loss + rle_loss
            else:
                base_loss += criterion_base(log_probs_base.transpose(0, 1), rle_bases, out_lengths / stride, rle_lengths)
                rle_loss += criterion_rle(log_probs_rle.transpose(0, 1), rle_rles, out_lengths / stride, rle_lengths)
                test_loss = base_loss + rle_loss

            predictions_bases.append(torch.exp(log_probs_base).cpu())
            predictions_rles.append(torch.exp(log_probs_rle).cpu())
            prediction_lengths.append(out_lengths / stride)

            progress_bar.refresh()
            progress_bar.update(1)
            progress_bar.set_description("BASE LOSS: " + str(base_loss.item()) + " RLE LOSS: " + str(rle_loss.item()))
    progress_bar.close()

    predictions_bases = np.concatenate(predictions_bases)
    predictions_rles = np.concatenate(predictions_rles)
    lengths = np.concatenate(prediction_lengths)

    references = [decode_ref(target, alphabet) for target in test_loader.dataset.targets]
    sequences = [decode_ctc_rle(post[:n], rle[:n], alphabet) for post, rle, n in zip(predictions_bases, predictions_rles, lengths)]

    if all(map(len, sequences)):
        accuracies = list(starmap(accuracy, zip(references, sequences)))
    else:
        accuracies = [0]

    mean = np.mean(accuracies)
    median = np.median(accuracies)
    sys.stderr.write(TextColor.GREEN + "BASE LOSS:              %.4f" % (base_loss / batch_idx) + "\n")
    sys.stderr.write(TextColor.GREEN + "RLE LOSS:              %.4f" % (rle_loss / batch_idx) + "\n")
    sys.stderr.write(TextColor.GREEN + "TOTAL LOSS:              %.4f" % (test_loss / batch_idx) + "\n")
    sys.stderr.write("Validation Accuracy (mean):   %.3f%%" % max(0, mean) + "\n")
    sys.stderr.write("Validation Accuracy (median): %.3f%%" % max(0, median) + "\n" + TextColor.END)

    return test_loss.item() / batch_idx, mean, median
