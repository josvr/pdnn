PDNN Fork
====

This is a fork of PDNN. Check the [project webpage](http://www.cs.cmu.edu/~ymiao/pdnntk.html) for all documentation.

This fork adds:

- Migrated from Python 2 to Python 3
- Migrated to Theano 0.8
- Added support for BloscPack dumped numpy arrays. BloscPack is typical 200 times faster and 3 times smaller than Pickle/Gzip based on 1 GB (250M float32) numpy array. BloscPack files are supposed to have *.blp and the associated label vector file *.blp.labels. See examples for an updated example for converting MNIST to Bloscpack, and DNN/CNN using it.  
