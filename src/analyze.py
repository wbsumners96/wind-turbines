import argparse
from load_data import load_data


parser = argparse.ArgumentParser(description='Slow dancing with wind ' \
	+ 'turbines.')
parser.add_argument('data_path', help='path to the directory in which ' \
	+ 'the data is located.')
parser.add_argument('--type', help='type of data to load (ARD or CAU).')

data = load_data(data_path, type)
