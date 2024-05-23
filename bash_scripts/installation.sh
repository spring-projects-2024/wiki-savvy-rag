#!/bin/bash

pip install -r requirements.txt
pip install -e .
git clone https://github.com/MattiaSC01/JEPA.git
cd JEPA
pip install -r requirements.txt
pip install -e .