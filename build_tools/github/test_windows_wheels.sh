#!/bin/bash

set -e
set -x

PYTHON_VERSION=$1

docker container run \
    --rm scikit-learn/minimal-windows \
    powershell -Command "python -c 'import erasplit; erasplit.show_versions()'"

docker container run \
    -e erasplit_SKIP_NETWORK_TESTS=1 \
    -e OMP_NUM_THREADS=2 \
    -e OPENBLAS_NUM_THREADS=2 \
    --rm scikit-learn/minimal-windows \
    powershell -Command "pytest --pyargs erasplit"
