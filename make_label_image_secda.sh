#!/bin/bash -x

curr_dir=$(pwd)

if [[ "$1" = "vm" ]]; then
    sh ./tensorflow/lite/tools/make/build_pynq_lib.sh pynq_vm
    mkdir -p ./out
    cp ./tensorflow/lite/tools/make/gen/pynq_vm_armv7l/bin/label_image ./out/label_image_secda_vm
    echo "sh build_pynq_lib.sh pynq_vm"
else
    sh ./tensorflow/lite/tools/make/build_pynq_lib.sh pynq_sa
    mkdir -p ./out
    cp ./tensorflow/lite/tools/make/gen/pynq_sa_armv7l/bin/label_image ./out/label_image_secda_sa
    echo "sh build_pynq_lib.sh pynq_sa"
fi