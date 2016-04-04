
import sys
import cPickle
import gzip

if __name__ == '__main__':

    f = gzip.open('mnist_py3k.pkl.gz', 'rb')
    (train_set_mat,train_set_vec), (valid_set_mat,valid_set_vec), (test_set_mat,test_set_vec) = cPickle.load(f)
    blosc_args=bp.BloscArgs(clevel=9)
    bp.pack_ndarray_file(train_set_mat, 'train.blp', chunk_size='100M', blosc_args=blosc_args)
	bp.pack_ndarray_file(train_set_vec, 'train.blp.labels', chunk_size='100M', blosc_args=blosc_args)
    bp.pack_ndarray_file(valid_set_mat, 'train.blp', chunk_size='100M', blosc_args=blosc_args)
	bp.pack_ndarray_file(valid_set_vec, 'train.blp.labels', chunk_size='100M', blosc_args=blosc_args)
    bp.pack_ndarray_file(test_set_mat, 'train.blp', chunk_size='100M', blosc_args=blosc_args)
	bp.pack_ndarray_file(test_set_vec, 'train.blp.labels', chunk_size='100M', blosc_args=blosc_args)

