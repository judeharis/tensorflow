#!/bin/bash -x

curr_dir=$(pwd)
cd ./tensorflow/lite/tools/make/
sh build_pynq_lib.sh 

echo $curr_dir
cd $curr_dir
mkdir ./out
cp ./tensorflow/lite/tools/make/gen/pynq_armv7l/bin/label_image ./out/label_image_secda