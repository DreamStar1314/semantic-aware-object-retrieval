import caffe
import numpy as np
import h5py
import cv2
from rmac import apply_rmac_aggregation

def vecpostproc(x,a=1):
    #print "haah",x.shape,(sum(x**2)).shape
    if a!=1:
        x = np.sign(a)*(abs(x) **a)
    x = np.nan_to_num(x / np.sqrt(sum(x ** 2)))
    where_are_inf = np.isinf(x)
    x[where_are_inf] = 0
    #x = np.where(abs(x) < 1e5, x, 0)
    return x
# init: 88.66/92.29(qe 5)
# def process_segRegion(X,segMap):
#     classification = np.unique(segMap, return_index=False)  # number of classification
#     #weights = np.ones((150, 1, 1))
#     background = [13,5,12,7,133,22,14,21,2,149]
#     #print X.shape,segMap.shape,'******************'
#     for element0 in background:
#         if element0 in classification:
#             region = np.where(segMap == element0)
#             # if len(region[0])/float(X.shape[1]*X.shape[1])<0.40 :
#             if element0 == 2:
#                X[:, region[0], region[1]] = X[:, region[0], region[1]] * 1.8
#             elif element0 == 149:
#                X[:, region[0], region[1]] = X[:, region[0], region[1]] * 4
#
#             else:
#                 X[:, region[0], region[1]] = 0
#     return X
def process_segRegion(X,segMap):
    classification = np.unique(segMap, return_index=False)  # number of classification
    #weights = [1.0 for i in range(150)]
    weights =   [0.08218574, 0.13617882, 0.04536155, 0.08067805, 0.06852638, 0.08091277,
 0.07419075, 0.08006714, 0.082518,   0.0890229,  0.0783328,  0.07341928,
 0.07349543, 0.08025899, 0.08069816, 0.0826464,  0.08551238, 0.07446633,
 0.08142496, 0.08081736, 0.08240292, 0.09248126, 0.08269563, 0.08103627,
 0.08269244, 0.08098029, 0.09228352, 0.0804473,  0.08145878, 0.08128513,
 0.08160798, 0.08197004, 0.08326937, 0.08306051, 0.07170372, 0.08095671,
 0.08117291, 0.08067721, 0.08524229, 0.08120766, 0.08131728, 0.08208995,
 0.08975817, 0.07756954, 0.08067259, 0.08166585, 0.08317468, 0.08095679,
 0.08541895, 0.08167865, 0.08182855, 0.08284704, 0.08224803, 0.08160848,
 0.08031823, 0.0811382,  0.08113406, 0.08243755, 0.08158094, 0.0868851,
 0.08248535, 0.08966345, 0.0807222,  0.08118144, 0.08146513, 0.08200058,
 0.08311998, 0.08152454, 0.08451263, 0.08134086, 0.08118486, 0.08110395,
 0.08102575, 0.08122579, 0.08121044, 0.08179986, 0.08204578, 0.08163316,
 0.08168801, 0.08110548, 0.0820091,  0.07950897, 0.0828419,  0.08277708,
 0.07318597, 0.08117589, 0.08068476, 0.08247229, 0.08167598, 0.08069969,
 0.07721151, 0.08162005, 0.08188135, 0.08189409, 0.08190686, 0.08020815,
 0.08159957, 0.08161938, 0.08123757, 0.08122018, 0.08114231, 0.08181847,
 0.0814904,  0.0818939,  0.0828329,  0.08145638, 0.08203487, 0.08145554,
 0.08272156, 0.0804969,  0.08193265, 0.08176655, 0.08096009, 0.07859987,
 0.08046508, 0.08148649, 0.08137634, 0.08116522, 0.08173241, 0.08097441,
 0.08128937, 0.08111195, 0.08107003, 0.08189771, 0.08123042, 0.0812342,
 0.08320816, 0.08174371, 0.08079634, 0.08120481, 0.08157451, 0.08142921,
 0.05859222, 0.08144367, 0.08125558, 0.08150738, 0.0811121 , 0.08140732,
 0.08120862, 0.07978085, 0.08214851, 0.08044466, 0.08120491, 0.08132192,
 0.08136424, 0.08139163, 0.08136221, 0.08145035, 0.08224584, 0.08058096]
    for element0 in classification:
        region = np.where(segMap == element0)
        X[:, region[0], region[1]] = X[:, region[0], region[1]] * weights[int(element0)-1]
    return X

def apply_whiten(feature_vector,dout=0):

    # h5f = h5py.File('/home/jking/plankton/imageRetrieval/0_segRegion_weightBox/ResNet_oxford_pca.h5', 'r')
    h5f = h5py.File('/home/jking/plankton/imageRetrieval/0_segRegion_weightBox/resNet101/resNet101_oxford_pca.h5', 'r')
    feature_mean = np.array(h5f['feature_mean'])
    eigvec = np.array(h5f['eigvec'])
    eigval = np.array(h5f['eigval'])
    if dout == 0:
        dout = feature_vector.shape[0]  # 512*20
    # return feature_vector.shape[0]*feature_mean.shape[0],1)
    #feature_vector_ = np.tile(feature_vector,(feature_mean.shape[0],1)) - np.tile(feature_mean,(feature_vector.shape[0],1)).transpose()
    feature_vector_ = feature_vector-feature_mean.reshape(feature_vector.shape[0],1)
    eigval_zero = []
    for element in list(eigval[0:dout]):
        if element<1e-5:
            eigval_zero.append(0)
        else:
            eigval_zero.append(element**(-0.5))
    eigval_zero = np.array(eigval_zero)
    # eigval_zero = eigval[0:dout]
    # eigval_zero = np.where(eigval_zero>1e-5,eigval_zero**(-0.5),0)
    feature_vector_ = np.dot (np.dot(np.diag(eigval_zero),np.transpose(eigvec[:, 0:dout])) , feature_vector_)

    return feature_vector_   # 512*20

class segLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 2, 'This layer can only have one bottom'
        assert len(top) == 1, 'This layer can only have one top'

    def reshape(self, bottom, top):
        tmp_shape = list(bottom[0].data.shape)
        top[0].reshape(*tmp_shape) # (1,512,20,30)

    def forward(self, bottom, top):
        # bottom[1].data.shape = (768,1024)
        tmp_shape = list(bottom[0].data.shape) # [1, 512, 24, 32]
        # print '*************tmp_shape *********:', tmp_shape , bottom[1].data.shape
        pool5_FM = bottom[0].data.reshape(tmp_shape[1],tmp_shape[2], tmp_shape[3])
        segMap = cv2.resize(bottom[1].data, (tmp_shape[3], tmp_shape[2]), interpolation=cv2.INTER_LINEAR) # 86.75
        #segMap = cv2.resize(bottom[1].data.T, (tmp_shape[3], tmp_shape[2]), interpolation=cv2.INTER_LINEAR) # 83.10
        output = process_segRegion(pool5_FM, segMap)
        top[0].data[:] = output.reshape(*tmp_shape)

    def backward(self, top, propagate_down, bottom):
        raise NotImplementedError(
            "Backward pass not supported with this implementation")

class roiPoolingLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'This layer can only have one bottom'
        assert len(top) == 1, 'This layer can only have one top'

    def reshape(self, bottom, top):
        tmp_shape = list(bottom[0].data.shape) #(1,512,23,27)
        tmp_shape[2] = 1
        tmp_shape[3] = 1
        top[0].reshape(*tmp_shape) # (1,512,1,1)

    def forward(self, bottom, top):
        tmp_shape = list(bottom[0].data.shape) # (1,512,23,27)
        tmp_shape[2] = 1
        tmp_shape[3] = 1
        input_data = np.squeeze(bottom[0].data) # (512,23,37)
        feature_matrix = np.array(apply_rmac_aggregation(input_data)) #  11 * 512
        feature_vector = vecpostproc(np.sum(vecpostproc(apply_whiten(vecpostproc(feature_matrix.transpose()))),1)) #apply_pca on paris_features , then l2??
        #print 'hahaha',feature_vector.shape,feature_matrix.shape,bottom[0].data.shape,input_data.shape
        top[0].data[:] = feature_vector.reshape(*tmp_shape)

    def backward(self, top, propagate_down, bottom):
        raise NotImplementedError(
            "Backward pass not supported with this implementation")


class resizeSegLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 2, 'This layer can only have one bottom'
        assert len(top) == 1, 'This layer can only have one top'

    def reshape(self, bottom, top):
        tmp_shape = list(bottom[0].data.shape)
        top[0].reshape(*tmp_shape)

    def forward(self, bottom, top):
        tmp_shape = list(bottom[0].data.shape)
        top[0].data[:] = cv2.resize(bottom[1].data,(1,tmp_shape[1],tmp_shape[2],tmp_shape[3]),interpolation=cv2.INTER_LINEAR)

    def backward(self, top, propagate_down, bottom):
        raise NotImplementedError(
            "Backward pass not supported with this implementation")