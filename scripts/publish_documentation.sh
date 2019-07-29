#!/bin/bash
set -x
set -e

BASE=${PWD}
ENV_DIR=${BASE}/flambe-docu-env_${BUILD_NUMBER}

virtualenv -p python3.6 ${ENV_DIR}
source ${ENV_DIR}/bin/activate

pip install .
pip install -r docs/requirements.txt

cd docs
make html
cd ..

pip install boto3
pip install tqdm

python scripts/deploy_documentation.py ${BUCKET_NAME}

deactivate

rm -Rf ${ENV_DIR}
unset ENV_DIR
unset BASE
