"""
Bonito Basecaller
"""

import sys
import time
import os
import h5py
from math import ceil
from glob import glob
from textwrap import wrap
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.util import load_model, decode_ctc
from bonito.TextColor import TextColor
import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from ont_fast5_api.fast5_interface import get_fast5_file


def med_mad(x, factor=1.4826):
    """
    Calculate signal median and median absolute deviation
    """
    med = np.median(x)
    mad = np.median(np.absolute(x - med)) * factor
    return med, mad


def trim(signal, window_size=40, threshold_factor=3.0, min_elements=3):

    med, mad = med_mad(signal[-(window_size*25):])
    threshold = med + mad * threshold_factor
    num_windows = len(signal) // window_size

    for pos in range(num_windows):

        start = pos * window_size
        end = start + window_size

        window = signal[start:end]

        if len(window[window > threshold]) > min_elements:
            if window[-1] > threshold:
                continue
            return end, len(signal)

    return 0, len(signal)


def preprocess(x, min_samples=1000):
    start, end = trim(x)
    # REVISIT: we can potentially trim all the signal if this goes wrong
    if end - start < min_samples:
        start = 0
        end = len(x)
        #sys.stderr.write("badly trimmed read\n")

    med, mad = med_mad(x[start:end])
    norm_signal = (x[start:end] - med) / mad
    return norm_signal


def check_fast5(fast5_filepath):
    try:
        test_open = h5py.File(fast5_filepath, mode="r")
    except:
        # tried to narrow down into specific error but h5py throws different errors in different versions of python
        return False
    test_open.close()
    return True


def get_raw_data(fast5_filepath):
    """
    Get the raw signal and read id from the fast5 files
    """
    with get_fast5_file(fast5_filepath, mode="r") as f5:
        for read_id in f5.get_read_ids():
            read = f5.get_read(read_id)
            raw_data = read.get_raw_data(scale=True)
            raw_data = preprocess(raw_data)
            yield read_id, raw_data


def window(data, size, stepsize=1, padded=False, axis=-1):
    """
    Segment data in `size` chunks with overlap
    """
    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def stitch(predictions, overlap):
    stitched = [predictions[0, 0:-overlap]]
    for i in range(1, predictions.shape[0] - 1): stitched.append(predictions[i][overlap:-overlap])
    stitched.append(predictions[-1][overlap:])
    return np.concatenate(stitched)


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

    return output_dir


def chunks(file_names, chunk_length):
    """Yield successive n-sized chunks from file_names."""
    chunks = []
    for i in range(0, len(file_names), chunk_length):
        chunks.append(file_names[i:i + chunk_length])
    return chunks


def basecall(args, input_files, device_id):
    sys.stderr.write(TextColor.GREEN + "INFO: LOADING MODEL ON DEVICE: " + str(device_id) + "\n" + TextColor.END)
    model, stride, alphabet = load_model(args.model, args.config, args.gpu_mode)
    torch.cuda.set_device(device_id)

    model = model.cuda()
    model.eval()
    model = DDP(model, device_ids=device_id)
    sys.stderr.write(TextColor.GREEN + "INFO: LOADED MODEL ON DEVICE: " + str(device_id) + "\n" + TextColor.END)

    output_directory = handle_output_directory(os.path.abspath(args.output_directory))
    fasta_file = open(output_directory + args.file_prefix + "_" + str(device_id) + ".fasta", 'w')

    num_reads = 0
    num_chunks = 0

    t0 = time.perf_counter()
    sys.stderr.write(TextColor.GREEN + "STARTING INFERENCE\n" + TextColor.END)
    sys.stderr.flush()

    for count, fast5 in enumerate(input_files):
        if not check_fast5(fast5):
            sys.stderr.write(TextColor.YELLOW + "\nWARNING: FAST5 FILE ERROR: " + fast5 + ". SKIPPING THIS FILE.\n" + TextColor.END)
            continue

        if count % 10 == 0 and count > 0:
            sys.stderr.write(TextColor.GREEN + "\nINFO: FINISHED PROCESSING: " + count + " FILES ON DEVICE: " + device_id + TextColor.END)

        for read_id, raw_data in get_raw_data(fast5):
            if len(raw_data) <= args.chunksize:
                chunks = np.expand_dims(raw_data, axis=0)
            else:
                chunks = window(raw_data, args.chunksize, stepsize=args.chunksize - args.overlap)

            chunks = np.expand_dims(chunks, axis=1)

            num_reads += 1
            num_chunks += chunks.shape[0]
            predictions = []

            with torch.no_grad():
                for i in range(ceil(len(chunks) / args.batchsize)):
                    batch = chunks[i*args.batchsize: (i+1)*args.batchsize]
                    tchunks = torch.tensor(batch)
                    if args.gpu_mode:
                        tchunks = tchunks.to(device_id)

                    probs = torch.exp(model(tchunks))
                    predictions.append(probs.cpu())

                predictions = np.concatenate(predictions)

                if len(predictions) > 1:
                    predictions = stitch(predictions, int(args.overlap / stride / 2))
                else:
                    predictions = np.squeeze(predictions, axis=0)

                sequence = decode_ctc(predictions, alphabet)

                if sequence is not None and len(sequence) > 0:
                    fasta_file.write(">"+str(read_id) + "\n")
                    fasta_file.write('\n'.join(wrap(sequence, 100)) + "\n")

    t1 = time.perf_counter()
    sys.stderr.write(TextColor.GREEN + "INFO: TOTAL READS: %s\n" % num_reads + TextColor.END)
    sys.stderr.write(TextColor.GREEN + "INFO: SAMPLES PER SECOND %.1E\n" % (num_chunks * args.chunksize / (t1 - t0)) + TextColor.END)


import torch.distributed as dist
from torch.multiprocessing import Process


def cleanup():
    dist.destroy_process_group()


def setup(device_id, total_gpus, args, input_files, basecall):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=device_id, world_size=total_gpus)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)
    basecall(args, input_files, device_id)
    cleanup()

def main(args):
    if args.distributed:
        # device ids
        sys.stderr.write(TextColor.GREEN + "INFO: DISTRIBUTED SETUP\n" + TextColor.END)
        total_gpu_devices = torch.cuda.device_count()

        sys.stderr.write(TextColor.GREEN + "INFO: TOTAL GPU AVAILABLE: " + str(total_gpu_devices) + "\n" + TextColor.END)

        # chunk the inputs
        input_files = glob("%s/*fast5" % args.reads_directory)
        chunk_length = int(len(input_files) / total_gpu_devices) + 1
        file_chunks = []
        for i in range(0, len(input_files), chunk_length):
            file_chunks.append(input_files[i:i + chunk_length])


        # run the processes
        processes = []
        for device_id in range(total_gpu_devices):
            p = Process(target=setup, args=(device_id, total_gpu_devices, args, file_chunks[device_id], basecall))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # # generate the dictionary in parallel
        # with concurrent.futures.ProcessPoolExecutor(max_workers=total_gpu_devices) as executor:
        #     futures = [executor.submit(basecall, args, chunk, device_id) for device_id, chunk in zip(device_ids, file_chunks)]
        #     for fut in concurrent.futures.as_completed(futures):
        #         if fut.exception() is None:
        #             d_id = fut.result()
        #         else:
        #             sys.stderr.write(TextColor.RED + "ERROR: " + str(fut.exception()) + TextColor.END + "\n")
        #         fut._result = None  # python issue 27144
    exit(0)




def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument(
        "-i",
        "--reads_directory",
        type=str,
        required=True,
        help="Path to the input directory containing fast5 files of reads."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Path to the input directory containing fast5 files of reads."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the input directory containing fast5 files of reads."
    )
    parser.add_argument(
        "-g",
        "--gpu_mode",
        default=False,
        action='store_true',
        help="If set then PyTorch will use GPUs for inference. CUDA required."
    )
    parser.add_argument(
        "-d",
        "--distributed",
        default=False,
        action='store_true',
        help="If set then it will try to spawn one model per GPU."
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        default="./bonito_outputs/",
        type=str,
        help="Path to output directory."
    )
    parser.add_argument(
        "-p",
        "--file_prefix",
        default="Reads_bonito",
        type=str,
        help="Prefix to use for the output file."
    )
    parser.add_argument(
        "-bs",
        "--batchsize",
        default=64,
        type=int,
        help="Batch size for inference."
    )
    parser.add_argument(
        "-ol",
        "--overlap",
        default=600,
        type=int,
        help="Overlap size for inference."
    )
    parser.add_argument(
        "-csz",
        "--chunksize",
        default=2000,
        type=int,
        help="Chunk size for inference."
    )
    return parser
