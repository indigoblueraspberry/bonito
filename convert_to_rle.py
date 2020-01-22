import h5py
import numpy as np
from argparse import ArgumentParser
import time

def main(args):
	starttime = time.time()
	ori_reads = h5py.File(args.original_file, 'r')['Reads']
	new_file = h5py.File(args.new_file, 'w')
	new_file.create_group('Reads')
	new_reads = new_file['Reads']

	ids = list(ori_reads.keys())
	size = len(ids)

	check = int(size/10)

	progress = 10
	id_no = 0

	print('Converting to RLE')
	for read_id in ids[:10]:
		ori_reads.copy(read_id, new_reads)
		segments = new_reads[read_id]['Ref_to_signal'][()]
		labels = new_reads[read_id]['Reference'][()]

		new_segments = [segments[0]]
		new_labels = []
		curr_label = labels[0]
		count = 0

		for i in range(1,len(labels)):
			if labels[i] > 3:
				print('Labels are not ACTG; current label = {}'.format(label[i]))
				break
			if labels[i] != curr_label or count >= args.length_limit-1:
				new_labels.append(curr_label*10+count)
				new_segments.append(segments[i])
				curr_label = labels[i]
				count = 0
			else:
				count += 1

		new_labels.append(curr_label*10+count)
		new_segments.append(segments[len(segments)-1])

		del new_reads[read_id]['Reference']
		del new_reads[read_id]['Ref_to_signal']

		new_reads[read_id]['Reference'] = np.asarray(new_labels)
		new_reads[read_id]['Ref_to_signal'] = np.asarray(new_segments)

		id_no += 1
		if id_no%check == 0:
			print('{}% completed'.format(progress))
			progress += 10

	print('100% Completed')
	print('Conversion took {} seconds'.format(time.time() - starttime))

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("original_file")
	parser.add_argument("new_file")
	parser.add_argument("--length_limit", default=10, type=int)
	main(parser.parse_args())