from flambe import log
import argparse


def my_script(arg1: int, arg2: str):
    """Test script"""
    for i in range(10):
        log('Dummy', float(i), global_step=i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg1', type=int)
    parser.add_argument('--arg2', type=str)
    args = parser.parse_args()
    my_script(args.arg1, args.arg2)
