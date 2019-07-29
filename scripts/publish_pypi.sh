#!/usr/bin/env bash

BASE=${PWD}
ENV_DIR=${BASE}/flambe-release-env_${BUILD_NUMBER}

set -x

virtualenv -p python3.6 ${ENV_DIR} --system-site-packages
source ${ENV_DIR}/bin/activate

TAG_VERSION=$(echo "${GIT_BRANCH}" | cut -d / -f 3)
echo $TAG_VERSION
echo "${GIT_BRANCH}"
echo "${TAG_VERSION}"

pip install twine

TAG_VERSION=${TAG_VERSION} python setup.py sdist bdist_wheel
twine upload --repository-url ${PYPI_REPO_URL} dist/*

deactivate

rm -Rf ${ENV_DIR}
unset ENV_DIR
unset BASE
