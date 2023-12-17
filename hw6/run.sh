#!/bin/bash

# part1
python spectral.py   --img image1.png --output output/img1/spectral/ratio_pick_2/      -c 2 --init pick -m ratio
python spectral.py   --img image1.png --output output/img1/spectral/normalized_pick_2/ -c 2 --init pick -m normalized
python ker_kmeans.py --img image1.png --output output/img1/kernal/pick_2/              -c 2 --init pick
python spectral.py   --img image2.png --output output/img2/spectral/ratio_pick_2/      -c 2 --init pick -m ratio
python spectral.py   --img image2.png --output output/img2/spectral/normalized_pick_2/ -c 2 --init pick -m normalized
python ker_kmeans.py --img image2.png --output output/img2/kernal/pick_2/              -c 2 --init pick
# part2
python spectral.py   --img image1.png --output output/img1/spectral/ratio_pick_3/      -c 3 --init pick -m ratio
python spectral.py   --img image1.png --output output/img1/spectral/normalized_pick_3/ -c 3 --init pick -m normalized
python ker_kmeans.py --img image1.png --output output/img1/kernal/pick_3/              -c 3 --init pick
python spectral.py   --img image2.png --output output/img2/spectral/ratio_pick_3/      -c 3 --init pick -m ratio
python spectral.py   --img image2.png --output output/img2/spectral/normalized_pick_3/ -c 3 --init pick -m normalized
python ker_kmeans.py --img image2.png --output output/img2/kernal/pick_3/              -c 3 --init pick

python spectral.py   --img image1.png --output output/img1/spectral/ratio_pick_4/      -c 4 --init pick -m ratio
python spectral.py   --img image1.png --output output/img1/spectral/normalized_pick_4/ -c 4 --init pick -m normalized
python ker_kmeans.py --img image1.png --output output/img1/kernal/pick_4/              -c 4 --init pick
python spectral.py   --img image2.png --output output/img2/spectral/ratio_pick_4/      -c 4 --init pick -m ratio
python spectral.py   --img image2.png --output output/img2/spectral/normalized_pick_4/ -c 4 --init pick -m normalized
python ker_kmeans.py --img image2.png --output output/img2/kernal/pick_4/              -c 4 --init pick
# part3

python spectral.py   --img image1.png --output output/img1/spectral/ratio_pp_3/      -c 3 --init kmeans++ -m ratio
python spectral.py   --img image1.png --output output/img1/spectral/normalized_pp_3/ -c 3 --init kmeans++ -m normalized
python ker_kmeans.py --img image1.png --output output/img1/kernal/pp_3/              -c 3 --init kmeans++
python spectral.py   --img image2.png --output output/img2/spectral/ratio_pp_3/      -c 3 --init kmeans++ -m ratio
python spectral.py   --img image2.png --output output/img2/spectral/normalized_pp_3/ -c 3 --init kmeans++ -m normalized
python ker_kmeans.py --img image2.png --output output/img2/kernal/p_3/              -c 3 --init kmeans++



