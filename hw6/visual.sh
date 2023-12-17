#!/bin/bash

python spectral.py --img image1.png --method normalized --clusters 2
python spectral.py --img image1.png --method normalized --clusters 3

python spectral.py --img image1.png --method ratio --clusters 2
python spectral.py --img image1.png --method ratio --clusters 3

python spectral.py --img image2.png --method normalized --clusters 2
python spectral.py --img image2.png --method normalized --clusters 3

python spectral.py --img image2.png --method ratio --clusters 2
python spectral.py --img image2.png --method ratio --clusters 3
