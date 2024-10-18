#!/bin/bash

echo "Installing project dependencies..."
pip install -r requirements.txt

echo "Installing project..."
pip install -e .

if [[ " $@ " =~ " --finetuning-dependencies " ]]; then
    echo "Installing finetuning dependencies..."
    git clone https://github.com/MattiaSC01/JEPA.git
    cd JEPA
    pip install -r requirements.txt
    pip install -e .
else
    echo "Skipping finetuning dependencies installation."
fi
