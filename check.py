import h5py
import numpy as np
import random
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def to_DNA(number, rle):
	d = {0:'A',1:'C',2:'G',3:'T'}
	if rle:
		count = int(number%10)+1
		DNA = int(number/10)
		if DNA == 10:
			print(number)
		return '{}\n{}'.format(count,d[DNA])
	else:
		return d[number]

def main(args):
# original_file = 'bonito-training-data-ori.hdf5'
# new_file = 'bonito-training-data.hdf5'

	ori_file = h5py.File(args.original_file, 'r')
	new_file = h5py.File(args.new_file, 'r')

	ori_dset = ori_file['Reads']
	new_dset = new_file['Reads']
	ids = list(ori_dset.keys())

	random.seed(args.seed)

	random_ids = random.sample(ids, args.trials)

	for random_id in random_ids:

		fig, ax = plt.subplots(figsize=(20,8))
		ax2 = plt.subplot(212)
		ax1 = plt.subplot(211, sharex = ax2)
		ax1.set_xticks([])
		ax1.set_yticks([])
		ax2.set_yticks([])

		signal = ori_dset[random_id]['Dacs'][()]
		start = random.randint(0, signal[len(signal)-1])
		length = args.signal_length

		ori_labels = ori_dset[random_id]['Reference'][()] 
		ori_segments = ori_dset[random_id]['Ref_to_signal'][()] 
		new_labels = new_dset[random_id]['Reference'][()] 
		new_segments = new_dset[random_id]['Ref_to_signal'][()] 

		ax1.plot(signal)
		ax2.plot(signal)

		idx = 0

		while new_segments[idx] < start:
			idx += 1

		start_idx = idx

		empty = True

		while new_segments[idx] < start+length:
			new_label = to_DNA(new_labels[idx], True)
			ax2.axvline(new_segments[idx], color='black')
			ax2.text((new_segments[idx]+new_segments[idx+1])/2, 0.9*(np.max(signal)-min(0,np.min(signal))), new_label, size=20, horizontalalignment='center', verticalalignment='center')
			ax2.set_title('RLE')
			idx += 1
			empty = False

		end_idx = idx

		idx = 0

		while ori_segments[idx] < start:
			idx += 1

		while ori_segments[idx] < start+length:
			if ori_segments[idx] >= new_segments[start_idx] and ori_segments[idx] <= new_segments[end_idx]:
				ori_label = to_DNA(ori_labels[idx], False)
				ax1.axvline(ori_segments[idx], color='black')
				ax1.text((ori_segments[idx]+ori_segments[idx+1])/2, 0.9*(np.max(signal)-min(0,np.min(signal))), ori_label, size=20, horizontalalignment='center', verticalalignment='center')
				ax1.set_title('Original')
			idx += 1

		
		if not empty:
			plt.xlim(new_segments[start_idx], new_segments[end_idx])	
			plt.savefig('Read_{}_from_{}_to_{}.png'.format(random_id, start, start+length))
		
		plt.clf()

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("original_file")
	parser.add_argument("new_file")
	parser.add_argument("--seed", default=25, type=int)
	parser.add_argument("--trials", default=5, type=int)
	parser.add_argument("--signal_length", default=100, type=int)
	main(parser.parse_args())