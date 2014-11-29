#!/bin/bash
write_dir=`pwd`
pushd .
cd $HOME/ift6085_data/rgbd_dataset_freiburg3_nostructure_texture_near_withloop_validation/rgb
convert -delay 10 -loop 0 *_red.png $write_dir/movie.gif
popd
