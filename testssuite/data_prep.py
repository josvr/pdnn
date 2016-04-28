import bloscpack as bp
import gzip
import pickle
import sys

if __name__ == '__main__':

    f = gzip.open('mnist_py3k.pkl.gz', 'rb')
    (train_set_mat,train_set_vec), (valid_set_mat,valid_set_vec), (test_set_mat,test_set_vec) = pickle.load(f)
    blosc_args=bp.BloscArgs(clevel=9)
    bp.pack_ndarray_file(train_set_mat[0:25000,:], 'train.part0.blp', chunk_size='100M', blosc_args=blosc_args)
    bp.pack_ndarray_file(train_set_vec[0:25000], 'train.part0.blp.labels', chunk_size='100M', blosc_args=blosc_args)
    bp.pack_ndarray_file(valid_set_mat[0:5000,:], 'valid.part0.blp', chunk_size='100M', blosc_args=blosc_args)
    bp.pack_ndarray_file(valid_set_vec[0:5000], 'valid.part0.blp.labels', chunk_size='100M', blosc_args=blosc_args)
    bp.pack_ndarray_file(test_set_mat[0:5000,:], 'test.part0.blp', chunk_size='100M', blosc_args=blosc_args)
    bp.pack_ndarray_file(test_set_vec[0:5000], 'test.part0.blp.labels', chunk_size='100M', blosc_args=blosc_args)

    bp.pack_ndarray_file(train_set_mat[25000:,:], 'train.part1.blp', chunk_size='100M', blosc_args=blosc_args)
    bp.pack_ndarray_file(train_set_vec[25000:], 'train.part1.blp.labels', chunk_size='100M', blosc_args=blosc_args)
    bp.pack_ndarray_file(valid_set_mat[5000:,:], 'valid.part1.blp', chunk_size='100M', blosc_args=blosc_args)
    bp.pack_ndarray_file(valid_set_vec[5000:], 'valid.part1.blp.labels', chunk_size='100M', blosc_args=blosc_args)
    bp.pack_ndarray_file(test_set_mat[5000:,:], 'test.part1.blp', chunk_size='100M', blosc_args=blosc_args)
    bp.pack_ndarray_file(test_set_vec[5000:], 'test.part1.blp.labels', chunk_size='100M', blosc_args=blosc_args)

