"""
Bonito model evaluator
"""


import time
import torch
import numpy as np
from itertools import starmap
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.training import ChunkDataSet
from bonito.util import init, load_data, load_model
from bonito.util import decode_ctc, decode_ref, accuracy, poa, print_alignment

from torch.utils.data import DataLoader


def main(args):

    poas = []
    init(args.seed, args.device)

    print("* loading data")
    testdata = ChunkDataSet(*load_data(limit=args.chunks, shuffle=args.shuffle))
    dataloader = DataLoader(testdata, batch_size=args.batchsize)

    for w in [int(i) for i in args.weights.split(',')]:

        print("* loading model", w)
        model = load_model(args.model_directory, args.device, weights=w)

        print("* calling")
        predictions = []
        t0 = time.perf_counter()

        for data, *_ in dataloader:
            with torch.no_grad():
                log_probs = model(data.to(args.device))
                predictions.append(log_probs.exp().cpu().numpy())

        duration = time.perf_counter() - t0

        references = [decode_ref(target, model.alphabet) for target in dataloader.dataset.targets]
        sequences = [decode_ctc(post, model.alphabet) for post in np.concatenate(predictions)]
        accuracies = list(starmap(accuracy, zip(references, sequences)))

        if args.poa: poas.append(sequences)

        print("* mean      %.2f%%" % np.mean(accuracies))
        print("* median    %.2f%%" % np.median(accuracies))
        print("* time      %.2f" % duration)
        print("* samples/s %.2E" % (args.chunks * data.shape[2] / duration))

    if args.poa:

        print("* doing poa")
        t0 = time.perf_counter()
        # group each sequence prediction per model together
        poas = [list(seq) for seq in zip(*poas)]

        consensuses = poa(poas)
        duration = time.perf_counter() - t0

        accuracies = list(starmap(accuracy, zip(references, consensuses)))

        print("* mean      %.2f%%" % np.mean(accuracies))
        print("* median    %.2f%%" % np.median(accuracies))
        print("* time      %.2f" % duration)


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=9, type=int)
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--chunks", default=500, type=int)
    parser.add_argument("--batchsize", default=100, type=int)
    parser.add_argument("--poa", action="store_true", default=False)
    parser.add_argument("--shuffle", action="store_true", default=True)
    return parser
