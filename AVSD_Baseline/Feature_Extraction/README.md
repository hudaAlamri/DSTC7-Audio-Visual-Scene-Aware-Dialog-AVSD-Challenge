
# Feature Extraction

## -I3D Feature Extraction
We provide the I3D feature extractor, which is in [TensorFlow](https://www.tensorflow.org/). Please make sure to install the relavent python packages before running the code.

Specifically, we feed 16 (default) video frames with temporal stride of 4 (default) into the I3D model, which is pretrained in the [Kinetics dataset](https://deepmind.com/research/open-source/open-source-datasets/kinetics/). And then, extract the CNN feature from the 'Mixed_5c' layer. In terms of the details for the I3D algorithm, please refer to https://github.com/deepmind/kinetics-i3d

In practice, please configure the setting in "extract_i3d_features.sh" and run it for extracting the feature automaticlly. 
