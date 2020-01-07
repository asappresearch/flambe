from pathlib import Path
import os

# A folder used to keep track of flambe runs
FLAMBE_GLOBAL_FOLDER = os.path.join(str(Path.home()), ".flambe")
FLAMBE_CLUSTER_DEFAULT = os.path.join(FLAMBE_GLOBAL_FOLDER, 'cluster.yaml')

# Deep Learning AMI (Ubuntu) Version 26.0
AWS_AMI = 'ami-02bd97932dabc037b'

# Pytorch AMI
GCP_AMI = 'projects/deeplearning-platform-release/global/images/family/pytorch-1-1-cpu'
