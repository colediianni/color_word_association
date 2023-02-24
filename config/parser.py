import argparse
from config.config import *

def parser():
    parser = argparse.ArgumentParser(description='Arguments')

    parser.add_argument('--dataset', '-d', choices=DATASETS,
                        help='author of the dataset you wish to use')
    parser.add_argument('--model-type', '-m', choices=MODELS, default='clip',
                        help='type of image-language model to use')
    parser.add_argument('--template', '-t', choices=TEMPLATES, default='all',
                        help='template set to use')
    
    # parser.add_argument('--normalization-type', '-n', choices=NORMALIZATION_METHODS, default='none',
    #                     help='image normalization method to apply to data')
    parser.add_argument('--device', choices = ["cpu", "cuda"],
                        default=DEVICE, help='device to use for pytorch')
    parser.add_argument('--seed', '-s', type=int, default=123,
                        help='random seed (default: 123)')

    # parser.add_argument('--output-dir', '-o', type=str,
    #                     default=OUTPUT_PATH, help='output directory path')

    args = parser.parse_args()
    return args