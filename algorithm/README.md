## Installation

**Prepare Python**
```shell
conda create --name presentera python=3.8

pip install -r requirements.txt
```

**Download PoseNet model**
```shell
MODEL_PROTO = 'https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt'
MODEL_DATA = 'http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel'

wget $MODEL_PROTO
wget MODEL_DATA
```