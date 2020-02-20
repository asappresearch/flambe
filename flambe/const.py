# flake8: noqa

from pathlib import Path
import os

# PYTHON
PYTHON_VERSION = '3.7'

# A folder used to keep track of flambe runs
FLAMBE_GLOBAL_FOLDER = os.path.join(str(Path.home()), ".flambe")
FLAMBE_CLUSTER_DEFAULT_FOLDER = os.path.join(FLAMBE_GLOBAL_FOLDER, 'clusters')
FLAMBE_CLUSTER_DEFAULT_CONFIG = os.path.join(FLAMBE_GLOBAL_FOLDER, 'cluster.yaml')

# SERIALIZATION 
STATE_DICT_DELIMETER = '.'
FLAMBE_SOURCE_KEY = '_flambe_source'
FLAMBE_CLASS_KEY = '_flambe_class'
FLAMBE_CONFIG_KEY = '_flambe_config'
FLAMBE_DIRECTORIES_KEY = '_flambe_directories'
FLAMBE_STASH_KEY = '_flambe_stash'
KEEP_VARS_KEY = 'keep_vars'
VERSION_KEY = '_flambe_version'
DEFAULT_PROTOCOL = 2  # For pickling


# MB limits for extension folder
MB = 2**20
WARN_LIMIT_MB = 100

# AWS Pytorch AMI
AWS_AMI = 'ami-0698bcaf8bd9ef56d'

# Pytorch AMI
GCP_AMI = 'projects/deeplearning-platform-release/global/images/family/pytorch-1-1-cpu'

# ASCII LOGOS

ASCII_LOGO = """

/$$$$$$$$ /$$                         /$$             /$
| $$_____/| $$                        | $$           /$
| $$      | $$  /$$$$$$  /$$$$$$/$$$$ | $$$$$$$   /$$$$$$
| $$$$$   | $$ |____  $$| $$_  $$_  $$| $$__  $$ /$$__  $$
| $$__/   | $$  /$$$$$$$| $$ \ $$ \ $$| $$  \ $$| $$$$$$$$
| $$      | $$ /$$__  $$| $$ | $$ | $$| $$  | $$| $$_____/
| $$      | $$|  $$$$$$$| $$ | $$ | $$| $$$$$$$/|  $$$$$$$
|__/      |__/ \_______/|__/ |__/ |__/|_______/  \_______/


"""

ASCII_LOGO_DEV = """

/$$$$$$$$ /$$                         /$$             /$
| $$_____/| $$                        | $$           /$
| $$      | $$  /$$$$$$  /$$$$$$/$$$$ | $$$$$$$   /$$$$$$
| $$$$$   | $$ |____  $$| $$_  $$_  $$| $$__  $$ /$$__  $$     _
| $$__/   | $$  /$$$$$$$| $$ \ $$ \ $$| $$  \ $$| $$$$$$$$    | |
| $$      | $$ /$$__  $$| $$ | $$ | $$| $$  | $$| $$_____/  __| | _____   __
| $$      | $$|  $$$$$$$| $$ | $$ | $$| $$$$$$$/|  $$$$$$$ / _` |/ _ \ \ / /
|__/      |__/ \_______/|__/ |__/ |__/|_______/  \_______/| (_| |  __/\ V /
                                                           \__,_|\___| \_/

"""
