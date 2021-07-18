import argparse
from srgan import SRGAN

def parse_args():
	parser = argparse.ArgumentParser(description="Training script for SRGAN")

	parser.add_argument(
		'Path',
		type=str,
		help="Path to training dataset",
	)

	parser.add_argument(
		'Epochs',
		type=int,
		help="Number of training epochs"
	)

	parser.add_argument(
		'-r',
		'--resolution',
		type=int,
		nargs=2,
		default=[64, 64],
		help="Low-resolution images resolution(height, width)"
	)

	parser.add_argument(
		'-s',
		'--scale',
		type=int,
		default=4,
		help="Up-scaling factor"
	)

	parser.add_argument(
		'-t',
		'--test',
		type=str,
		default=None,
		help="Path to a test dataset"
	)

	parser.add_argument(
		'-w',
		'--weights',
		type=str,
		default="./weights/",
		help="Path to save a generator/discriminator weights"
	)

	return parser.parse_args()

if __name__ == '__main__':
	
	args = parse_args()
	
	srgan = SRGAN(
		lr_height=args.resolution[0],
		lr_width=args.resolution[1],
		channels=3,
		upscaling_factor=args.scale,
		training=True
	)

	srgan.train(
		epochs=args.Epochs,
		datapath=args.Path,
		test_datapath=args.test,
		batch_size=1,
		weights_path=args.weights
	)
