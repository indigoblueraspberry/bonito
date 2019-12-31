#!/bin/bash

num_chunks=1000000
url="https://storage.googleapis.com/kishwar-helen/bonito_data/bonito-training-data.hdf5"
outdir="bonito/data"
outfile="${outdir}/bonito-training-data.hdf5"

wget -q --show-progress --max-redirect=9 -O "$outfile" "$url"
./scripts/convert-data-rle.py "$outfile" "$outdir" --chunks "$num_chunks"
