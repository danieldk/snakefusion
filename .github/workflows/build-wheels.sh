#!/bin/sh

set -euo pipefail

cd /root

curl --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain stable -y
source $HOME/.cargo/env

PYROOT=/opt/python/cp36-cp36m
PYBIN=${PYROOT}/bin

${PYBIN}/pip install maturin auditwheel

${PYBIN}/pip install maturin
${PYBIN}/maturin build -i ${PYBIN}/python \
  --release \
  --compatibility manylinux2014

for wheel in target/wheels/*.whl; do
    ${PYBIN}/auditwheel repair "${wheel}"
done
