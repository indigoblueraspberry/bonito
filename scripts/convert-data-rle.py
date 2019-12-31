#!/usr/bin/env python

"""
Convert a chunkify training file
"""

import os
import h5py
import random
import numpy as np
from bisect import bisect_left
from argparse import ArgumentParser
from itertools import chain, zip_longest


def align(samples, pointers):
    """ align to the start of the mapping """
    return samples[pointers[0]:pointers[-1]], pointers - pointers[0]


def scale(read, samples, normalise=True):
    """ scale and normalise a read """
    scaling = read.attrs['range'] / read.attrs['digitisation']
    scaled = (scaling * (samples + read.attrs['offset'])).astype(np.float32)
    if normalise:
        return (scaled - read.attrs['shift_frompA']) / read.attrs['scale_frompA']
    return scaled


def num_reads(tfile):
    """ return the sample lengths for each read in the training file """
    with h5py.File(tfile, 'r') as training_file:
        return len(training_file['Reads'])


def get_reads(tfile):
    """ get each dataset per read """
    with h5py.File(tfile, 'r') as training_file:
        for read_id in np.random.permutation(training_file['Reads']):
            if read_id != '6b39994b-80ef-4068-a3bc-a3e38931bee8':
                continue

            read = training_file['Reads/%s' % read_id]
            reference = read['Reference'][:]
            pointers = read['Ref_to_signal'][:]
            samples = read['Dacs'][:]
            samples = scale(read, samples)
            samples, pointers = align(samples, pointers)
            yield read_id, samples, reference, pointers


def encode_to_rle(targets, target_length, target_rle_base, target_rles):
    running_base = targets[0]
    running_rle = 1
    target_rle_length = 0

    for i in range(1, target_length):
        current_base = targets[i]
        if running_base == current_base and running_base != 0:
            running_rle += 1
        else:
            target_rle_base[target_rle_length] = running_base
            target_rles[target_rle_length] = running_rle

            target_rle_length += 1
            running_base = current_base
            running_rle = 1

    target_rle_base[target_rle_length] = running_base
    target_rles[target_rle_length] = running_rle
    target_rle_length += 1

    return target_rle_base, target_rles, target_rle_length


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_directory, exist_ok=True)

    read_idx = 0
    chunk_idx = 0
    chunk_count = 0

    min_bases = 0
    max_bases = 0
    off_the_end_ref = 0
    off_the_end_sig = 0

    total_reads = num_reads(args.chunkify_file)

    chunks = np.zeros((args.chunks, args.max_seq_len * args.max_samples_per_base), dtype=np.float32)
    chunk_lengths = np.zeros(args.chunks, dtype=np.uint16)

    targets = np.zeros((args.chunks, args.max_seq_len), dtype=np.uint8)
    target_lengths = np.zeros(args.chunks, dtype=np.uint16)

    targets_rle_base = np.zeros((args.chunks, args.max_seq_len), dtype=np.uint8)
    targets_rles = np.zeros((args.chunks, args.max_seq_len), dtype=np.uint8)
    targets_rle_size = np.zeros(args.chunks, dtype=np.uint16)

    for read_id, samples, reference, pointers in get_reads(args.chunkify_file):

        read_idx += 1

        sequence_length = len(reference)
        squiggle_duration = len(samples)

        # overlap chunks with +/- 2.5% of max seq len
        overlap = np.int32(args.max_seq_len * 0.025)

        # first chunk
        seq_starts = 0
        seq_ends = np.random.randint(args.min_seq_len, args.max_seq_len)

        chunk_idxs = [(seq_starts, seq_ends)]

        # variable size chunks with overlap
        while seq_ends < sequence_length - args.min_seq_len:
            seq_starts = seq_ends + np.random.randint(-overlap, overlap)
            seq_ends = seq_starts + np.random.randint(args.min_seq_len, args.max_seq_len)
            seq_ends = min(seq_ends, sequence_length)
            chunk_idxs.append((seq_starts, seq_ends))

        for start, end in chunk_idxs:

            chunk_idx += 1

            if end > sequence_length:
                off_the_end_ref += 1
                continue

            squiggle_start = pointers[start]
            squiggle_end = pointers[end]

            squiggle_length = squiggle_end - squiggle_start

            reference_length = end - start

            samples_per_base = squiggle_length / reference_length

            if samples_per_base < args.min_samples_per_base:
                min_bases += 1
                continue

            if samples_per_base > args.max_samples_per_base:
                max_bases += 1
                continue

            if squiggle_end > squiggle_duration:
                off_the_end_sig += 1
                continue

            chunks[chunk_count, :squiggle_length] = samples[squiggle_start:squiggle_end]
            chunk_lengths[chunk_count] = squiggle_length

            # index alphabet from 1 (ctc blank labels - 0)
            targets[chunk_count, :reference_length] = reference[start:end] + 1
            target_lengths[chunk_count] = reference_length

            # print(squiggle_start, squiggle_end)
            # print("Chunks:", chunks[chunk_count])
            # print("REFERENCE LENGTH: ", target_lengths[chunk_count])
            targets_rle_base[chunk_count], targets_rles[chunk_count], targets_rle_size[chunk_count] = \
                encode_to_rle(targets[chunk_count], target_lengths[chunk_count],
                              targets_rle_base[chunk_count], targets_rles[chunk_count])

            # print("Targets")
            # labels = ["N", "T", "G", "C", "A"]
            # for t in targets[chunk_count]:
            #     print(labels[t], end='')
            # print()
            # for t in targets_rle_base[chunk_count]:
            #     print(labels[t], end='')
            # print()
            # for t in targets_rles[chunk_count]:
            #     print(t, end='')
            # print()
            # exit()
            chunk_count += 1

            if chunk_count == args.chunks:
                break

        if chunk_count == args.chunks:
            break

    skipped = chunk_idx - chunk_count
    percent = (skipped / chunk_count * 100) if skipped else 0

    print("Processed %s reads of out %s [%.2f%%]" % (read_idx, total_reads, read_idx / total_reads * 100))
    print("Skipped %s chunks out of %s due to bad chunks [%.2f%%].\n" % (skipped, chunk_count, percent))
    print("Reason for skipping:")
    print("  - off the end (signal)          ", off_the_end_sig)
    print("  - off the end (sequence)        ", off_the_end_ref)
    print("  - minimum number of bases       ", min_bases)
    print("  - maximum number of bases       ", max_bases)

    if chunk_count < args.chunks:
        chunks = np.delete(chunks, np.s_[chunk_count:], axis=0)
        chunk_lengths = chunk_lengths[:chunk_count]
        targets = np.delete(targets, np.s_[chunk_count:], axis=0)
        target_lengths = target_lengths[:chunk_count]
        targets_rle_base = np.delete(targets_rle_base, np.s_[chunk_count:], axis=0)
        targets_rles = np.delete(targets_rles, np.s_[chunk_count:], axis=0)
        targets_rle_size = targets_rle_size[:chunk_count]

    np.save(os.path.join(args.output_directory, "chunks.npy"), chunks)
    np.save(os.path.join(args.output_directory, "chunk_lengths.npy"), chunk_lengths)
    np.save(os.path.join(args.output_directory, "references.npy"), targets)
    np.save(os.path.join(args.output_directory, "reference_lengths.npy"), target_lengths)
    np.save(os.path.join(args.output_directory, "rle_reference_bases.npy"), targets_rle_base)
    np.save(os.path.join(args.output_directory, "rle_reference_rles.npy"), targets_rles)
    np.save(os.path.join(args.output_directory, "rle_reference_lengths.npy"), targets_rle_size)

    print()
    print("Training data written to %s:" % args.output_directory)
    print("  - chunks.npy with shape", chunks.shape)
    print("  - chunk_lengths.npy with shape", chunk_lengths.shape)
    print("  - references.npy with shape", targets.shape)
    print("  - reference_lengths.npy shape", target_lengths.shape)
    print("  - rle_reference_bases.npy shape", targets_rle_base.shape)
    print("  - rle_reference_rles.npy shape", targets_rles.shape)
    print("  - rle_reference_lengths.npy shape", targets_rle_size.shape)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("chunkify_file")
    parser.add_argument("output_directory")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--chunks", default=2000000, type=int)
    parser.add_argument("--min-seq-len", default=256, type=int)
    parser.add_argument("--max-seq-len", default=512, type=int)
    parser.add_argument("--min-samples-per-base", default=8, type=int)
    parser.add_argument("--max-samples-per-base", default=16, type=int)
    main(parser.parse_args())
