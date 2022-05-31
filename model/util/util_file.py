import os
import sys


def check_path(path):
    if not os.path.exists(path):
        print('Not found {}'.format(path))
        sys.exit(0)


def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)