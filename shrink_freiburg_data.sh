#!/bin/bash
pushd .
cd $HOME/ift6085_data/rgbd_dataset_freiburg3_nostructure_texture_near_withloop_validation/rgb
# Make sure the old reductions are gone
rm *_red.png
#Make sure the convert program is installed sudo apt-get install imagemagick
#for i in $(ls *.png | sed -n '0~10p'); do convert $i -colorspace Gray -resize 28x28\! "${i%.png}_red.png"; done
for i in `ls *.png`; do convert $i -colorspace Gray -resize 28x28\! "${i%.png}_red.png"; done
popd
