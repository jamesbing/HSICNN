#!/usr/bin/env sh
set -e

/home/para/caffe/build/tools/caffe train --solver=/home/para/gitstore/HSICNN/vgg/solver.prototxt $@
