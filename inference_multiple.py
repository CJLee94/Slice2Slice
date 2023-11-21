import os, glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default=None, help="the directory contains files that need to be processed")
    parser.add_argument("--test_ckpt", default=None, type=str)
    parser.add_argument("--direction", default="Ascan")
    return parser.parse_args()


def main():
    args = parse_args()
    for f in glob.glob(os.path.join(args.data_dir, "*")):
        os.system("python inference.py -d {} --test_ckpt {} --direction {}".format(f, args.test_ckpt, args.direction))

if __name__=="__main__":
    main()