PDNN Fork
====

This is a fork of PDNN. Check the [project webpage](http://www.cs.cmu.edu/~ymiao/pdnntk.html) for all documentation.

I created this fork for for my Master Thesis project about Deep Learning. This Fork is now mainly focussed on DNN/SDA. 

Here is a list of changes I made until so far: 

- Migrated from Python 2 to Python 3
- Migrated to Theano 0.8
- Added support for BloscPack dumped numpy arrays. **BloscPack is typical 200 times faster and 3 times smaller than Pickle/Gzip based on 1 GB (250M float32) numpy array**. BloscPack files are supposed to have *.blp and the associated label vector file *.blp.labels. See examples for an updated example for converting MNIST to Bloscpack, and DNN/CNN using it.  
- SDA/DNN/CNN arguments are now dumped to stderr (for checking what is passed)
- SDA intermediate output tmp save path/filename is now different from DNN. This avoids temp state of SDA (in case SDA crashes) is then used for DNN fine tuning. 
- Code saving intermediate state (model save step param) for Neural Network (file2nnet, nnet2file) is using High performance BloscPack instead of (slow) JSON. Because Bloscpack can only save 1 array in 1 file (to my knowledge, let me know if this is possible), it now saves data in a directory.