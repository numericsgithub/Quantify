#!/bin/bash
# scripts/test.sh

export QUANT_WORKDIR=/home/th/tmp/quanttests

cd /home/th/Desktop/brevitas-quantizers
pip install -e . -q 2>/dev/null
pip install -r requirements.txt -q 2>/dev/null
pytest tests/ -v
