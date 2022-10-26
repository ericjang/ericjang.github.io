---
layout: post
title:  "A Tutorial on Webdataset"
date:   2022-08-27
categories:
summary: 
---

This can turn into a book chapter on data loading.

Simple solution - indexed dataset with file-backed can scale to 100Gb+ datasets on SSD, and remain quite performant compared to RAM.

For tiny datasets, You can mount ramfs and copy the files to RAM as well.

As the dataset gets larger, you start to need to switch to a streaming format.



I recently experimented with several data loader options for PyTorch: Torch's built-in `ImageFolder`, [FFCV](https://github.com/libffcv/ffcv/tree/main/ffcv), and [WebDataset](https://github.com/webdataset/webdataset).

Dataset|Pros|Cons
ImageFolder|Simple to get started, can thumb through images quickly to find featurization bugs|Inflexible modeling
FFCV|Screaming fast|Inflexible modeling
WebDataset|Handles large scale datasets well, flexible modeling|Takes some work to get statistically uniform sampling


Being able to train models in 3 hours vs. 6 hours is like night and day in terms of what creative ideas you can try.


# You Must Learn I.I.D

Yoshua Bengio has often mentioned that it feels aesthetically wrong that our learning algorithms need to learn I.I.D, whereas humans can handle learning sequentially. That may be true, and for these cases you want a sequential model like a transformer to learn adaptive behavior, or a fancier meta-learning approach.

But if you are training a neural net from scratch, it makes absolutely no sense to

But the way you train the sequence model / meta-learner *must* be i.i.d, unless it was already imbued with the priors needed to learn sequentially in a few-shot manner. Otherwise, 



Some good links on how to structure data loaders:
https://webdataset.github.io/webdataset/multinode/

https://github.com/webdataset/webdataset/issues/71#issuecomment-843397824
https://github.com/webdataset/webdataset/issues/47#issuecomment-800019973

https://docs.ffcv.io/benchmarks.html


```
from PIL import features
features.check_feature("libjpeg_turbo")  
```


If worker tries to open an existing file, it will complain
venv/lib/python3.8/site-packages/webdataset/pipeline.py:69: ResourceWarning: unclosed file <_io.BufferedReader name='/home/eric/datasets/featurized_data/1.tar'>

files that are already open are really bad for I/O because disk has to seek to that spot again!

so it's ideal to re-shard data:

https://tarproc.readthedocs.io/en/latest/examples/

tarscat testdata/sample.tar testdata/sample.tar | tarsplit -n 32


installing tarp from source:
```
sudo apt-get install libczmq-dev
git clone https://github.com/tmbdev/tarp.git
cd tarp
make bin/tarp
sudo make install
```

if your data is relatively small, consider memory-mapping it so all the data is in RAM! 