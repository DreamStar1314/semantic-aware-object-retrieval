#-*-coding:utf-8-*-
import sys
import numpy as np
import caffe
import cv2
from tqdm import tqdm
import os
from collections import OrderedDict
import subprocess
import h5py

class Parameters:
    def __init__(self):
        #python main_softAtt.py --gpu 0 --S 800 --L 2 --proto train_vgg_tripletNet.prototxt --weights model.caffemodel --dataset datasets/Paris --eval_binary datasets/evaluation/compute_ap --temp_dir tmp_PLM --dataset_name Paris --multires  --aqe 1 --dbe 20
        self.gpu = 0
        self.ratio = 1.0
        self.L = 3
        # self.proto = 'test_resnet101_rmac.prototxt'
        # self.weights = 'model.caffemodel'
        self.proto = 'deploy_resnet101_normpython.prototxt'
        self.weights = '/home/jking/plankton/imageRetrieval/0_segRegion_weightBox/resNet101/model.caffemodel'
        self.dataset = '/home/jking/plankton/imageRetrieval/cnn-cbir-benchmark/data/paris6k'
        self.eval_binary = '/home/jking/plankton/imageRetrieval/3_DeepRetrieval/evaluation/compute_ap'
        self.temp_dir = 'tmp_GCN'
        self.dataset_name = 'Paris'
        self.multires = False #False
        self.aqe = 15 # 1
        self.dbe = 20 # 20

class ImageHelper:
    def __init__(self, ratio, L, means):
        self.ratio = ratio # 1.0
        self.L = L # 2
        self.means = means # shape = (1, 3, 1, 1)

    def prepare_image_and_grid_regions_for_network(self, fname, imgSign,roi=None):
        # Extract image, resize at desired size, and extract roi region if
        # available. Then compute the rmac grid in the net format: ID X Y W H
        # 直接从原图crop后的图像，按照要求resize(the same ratio for h/w) im_resized， 再交换通道，且减去均值 I；
        # I : (1, 3, 715, 713)
        # im_resized : (715, 713, 3)
        # segMap : (715, 713)
        I, segMap = self.load_and_prepare_image(fname, imgSign, roi)

        return I, segMap

    def get_rmac_features(self, I, segMap,net):
        net.blobs['data'].reshape(1, 2048, int(I.shape[1]), int(I.shape[2]))
        net.blobs['data'].data[:] = I[np.newaxis,:,:,:]
        net.blobs['segMap'].reshape(segMap.shape[0], segMap.shape[1])
        net.blobs['segMap'].data[:] = segMap
        #print '***********  get_rmac_features ****************',I.shape,R.shape,segMap.shape
        # net.forward(end='rmac/normalized')
        # return np.squeeze(net.blobs['rmac/normalized'].data)
        net.forward(end='roiPooling0')
        return np.squeeze(net.blobs['roiPooling0'].data)

    def load_and_prepare_image(self, fname, imgSign,roi=None):
        # Read image, get aspect ratio, and resize such as the largest side equals S
        segImg = cv2.imread('/home/jking/plankton/imageRetrieval/cnn-cbir-benchmark/data/flick100k_seg/gray/' + os.path.basename(fname).replace('.jpg','.png'),0)
        ind = fname.split('/')[-2]
        if ind == 'paris' or ind == 'paris6k':
            h5f_pool5 = h5py.File('/home/jking/plankton/imageRetrieval/0_segRegion_weightBox/get_pool5_data/paris_​res5cReLu_features.h5','r')
            fname = '/home/jking/plankton/imageRetrieval/2_rmac/datasets/paris6k/' + os.path.basename(fname)
        elif ind == 'flickr100k_clean':
            h5f_pool5 = h5py.File('/home/jking/plankton/imageRetrieval/0_segRegion_weightBox/get_pool5_data/flick100k_​res5cReLu_features.h5','r')
        pool5_data = np.array(h5f_pool5[fname])
        ratio = self.ratio

        # segMap: (715, 713)
        segMap = cv2.resize(segImg,(np.round(segImg.shape[1]*ratio).astype(np.int32), np.round(segImg.shape[0]*ratio).astype(np.int32)))
        #print '*****',segMap.shape, pool5_data.shape
        return pool5_data, segMap

class Dataset:
    def __init__(self, path, eval_binary_path):
        self.path = path # 'datasets/Paris'
        self.eval_binary_path = eval_binary_path #'datasets/evaluation/compute_ap'
        # Some images from the Paris dataset are corrupted. Standard practice is
        # to ignore them
        self.blacklisted = set(["paris_louvre_000136",
                            "paris_louvre_000146",
                            "paris_moulinrouge_000422",
                            "paris_museedorsay_001059",
                            "paris_notredame_000188",
                            "paris_pantheon_000284",
                            "paris_pantheon_000960",
                            "paris_pantheon_000974",
                            "paris_pompidou_000195",
                            "paris_pompidou_000196",
                            "paris_pompidou_000201",
                            "paris_pompidou_000467",
                            "paris_pompidou_000640",
                            "paris_sacrecoeur_000299",
                            "paris_sacrecoeur_000330",
                            "paris_sacrecoeur_000353",
                            "paris_triomphe_000662",
                            "paris_triomphe_000833",
                            "paris_triomphe_000863",
                            "paris_triomphe_000867"]) #20
        self.load()

    def load(self):
        # Load the dataset GT
        self.lab_root = '{0}/gt_files/'.format(self.path) #'datasets/Paris/lab/'
        self.img_root = '{0}/paris/'.format(self.path) #'datasets/Paris/jpg/'
        lab_filenames = np.sort(os.listdir(self.lab_root))
        # Get the filenames without the extension, 6392 'paris_defense_000000'
        self.oxford_blocks = open('/home/jking/plankton/imageRetrieval/0_segRegion_weightBox/get_pool5_data/paris106k_imlist.txt','r').read().splitlines()
        self.img_filenames = [os.path.basename(e).split('.')[0] for e in self.oxford_blocks]

        # Parse the label files. Some challenges as filenames do not correspond
        # exactly to query names. Go through all the labels to:
        # i) map names to filenames and vice versa
        # ii) get the relevant regions of interest of the queries,
        # iii) get the indexes of the dataset images that are queries
        # iv) get the relevants / non-relevants list
        self.relevants = {} #dict = {query: list } len(dict) = 55, query = 'defense_1',each list contains the all id number of relevant images
        self.junk = {} # dict = {query: list } len(dict) = 55, query = 'defense_1',each list contains the all id number of junk images
        self.non_relevants = {} # dict = {query: list } len(dict) = 55, each list contains the all id number of non-relevant images

        self.filename_to_name = {} # 55 'paris_defense_000038': triomphe_5, 'paris_museedorsay_000012': museedorsay_2
        self.name_to_filename = OrderedDict()
        self.q_roi = {} # dict,55, the query ROI,'defense_1' = [ 130.   17.  843.  732.]
        for e in lab_filenames:
            if e.endswith('_query.txt'):
                q_name = e[:-len('_query.txt')]
                q_data = file("{0}/{1}".format(self.lab_root, e)).readline().split(" ")
                q_filename = q_data[0][5:] if q_data[0].startswith('oxc1_') else q_data[0]
                self.filename_to_name[q_filename] = q_name
                self.name_to_filename[q_name] = q_filename
                good = set([e.strip() for e in file("{0}/{1}_ok.txt".format(self.lab_root, q_name))])
                good = good.union(set([e.strip() for e in file("{0}/{1}_good.txt".format(self.lab_root, q_name))]))
                junk = set([e.strip() for e in file("{0}/{1}_junk.txt".format(self.lab_root, q_name))])
                good_plus_junk = good.union(junk)
                self.relevants[q_name] = [i for i in range(len(self.img_filenames)) if self.img_filenames[i] in good]
                self.junk[q_name] = [i for i in range(len(self.img_filenames)) if self.img_filenames[i] in junk]
                self.non_relevants[q_name] = [i for i in range(len(self.img_filenames)) if self.img_filenames[i] not in good_plus_junk]
                self.q_roi[q_name] = np.array(map(float, q_data[1:]), dtype=np.float32)

        self.q_names = self.name_to_filename.keys() # query name ,55, 'defense_1'
        self.q_index = np.array([self.img_filenames.index(self.name_to_filename[qn]) for qn in self.q_names]) # 55,the query index in dataset
        self.N_images = len(self.img_filenames) # 6392
        self.N_queries = len(self.q_index) # 55

    def score(self, sim, temp_dir, eval_bin):
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        idx = np.argsort(sim, axis=1)[:, ::-1]
        maps = [self.score_rnk_partial(i, idx[i], temp_dir, eval_bin) for i in range(len(self.q_names))]
        for i in range(len(self.q_names)):
            print "{0}: {1:.2f}".format(self.q_names[i], 100 * maps[i])
        print 20 * "-"
        print "Mean: {0:.2f}".format(100 * np.mean(maps))

    def score_rnk_partial(self, i, idx, temp_dir, eval_bin):
        rnk = np.array(self.img_filenames)[idx]
        with open("{0}/{1}.rnk".format(temp_dir, self.q_names[i]), 'w') as f:
            f.write("\n".join(rnk)+"\n")
        cmd = "{0} {1}{2} {3}/{4}.rnk".format(eval_bin, self.lab_root, self.q_names[i], temp_dir, self.q_names[i])
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        map_ = float(p.stdout.readlines()[0])
        p.wait()
        return map_

    def get_filename(self, i):
        return os.path.normpath("{0}/{1}.jpg".format(self.img_root, self.img_filenames[i]))

    def get_query_filename(self, i):
        return os.path.normpath("{0}/{1}.jpg".format(self.img_root, self.img_filenames[self.q_index[i]]))

    def get_query_roi(self, i):
        return self.q_roi[self.q_names[i]]


def extract_features(dataset, image_helper, net, parameters):
    ratios = [parameters.ratio, ] if not parameters.multires else [ 1,0.8, 0.6]
    # First part, queries
    for ratio in ratios:
        # Set the scale of the image helper
        image_helper.ratio = ratio
        # 'tmp_PLM/Paris_S800_L2_dataset.npy'
        out_queries_fname = "{0}/{1}_S{2}_L{3}_queries.npy".format(parameters.temp_dir, parameters.dataset_name, int(ratio*10), parameters.L)
        if not os.path.exists(out_queries_fname):
            dim_features = net.blobs['roiPooling0'].data.shape[1] # 1*2048
            N_queries = dataset.N_queries # 55
            features_queries = np.zeros((N_queries, dim_features), dtype=np.float32) # 55*2048
            for i in tqdm(range(N_queries), file=sys.stdout, leave=False, dynamic_ncols=True):
                # Load image, process image, get image regions, feed into the network, get descriptor, and store
                # I: 1*3*461*487 直接从原图crop后的图像，按照要求resize(teh same ratio for h/w)， 再交换通道，且减去均值
                # R: 8*5 the region in the initial image , not the feature map
                I, segMap = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_query_filename(i), 'paris_query',roi=dataset.get_query_roi(i), )
                # input the image and rois into the DCNN , the the feature vectors
                features_queries[i] = image_helper.get_rmac_features(I, segMap, net) # 55*2048
            np.save(out_queries_fname, features_queries)
    # 55*2048
    features_queries = np.dstack([np.load("{0}/{1}_S{2}_L{3}_queries.npy".format(parameters.temp_dir, parameters.dataset_name, int(ratio*10), parameters.L)) for ratio in ratios]).sum(axis=2)

    features_queries /= np.sqrt((features_queries * features_queries).sum(axis=1))[:, None]

    # Second part, dataset
    for ratio in ratios:
        image_helper.ratio = ratio
        out_dataset_fname = "{0}/{1}_S{2}_L{3}_dataset.npy".format(parameters.temp_dir, parameters.dataset_name, int(ratio*10), parameters.L)
        if not os.path.exists(out_dataset_fname):
            dim_features = net.blobs['roiPooling0'].data.shape[1]
            N_dataset = dataset.N_images
            features_dataset = np.zeros((N_dataset, dim_features), dtype=np.float32)
            for i in tqdm(range(N_dataset), file=sys.stdout, leave=False, dynamic_ncols=True):
                # Load image, process image, get image regions, feed into the network, get descriptor, and store
                I, segMap = image_helper.prepare_image_and_grid_regions_for_network(dataset.oxford_blocks[i], 'paris_dataset', roi=None)
                features_dataset[i] = image_helper.get_rmac_features(I, segMap,net)
            np.save(out_dataset_fname, features_dataset)
    features_dataset = np.dstack([np.load("{0}/{1}_S{2}_L{3}_dataset.npy".format(parameters.temp_dir, parameters.dataset_name, int(ratio*10), parameters.L)) for ratio in ratios]).sum(axis=2)
    # np.sqrt((features_dataset * features_dataset).sum(axis=1))[:, None] , 6392*1; features_dataset: 6392*2048
    features_dataset /= np.sqrt((features_dataset * features_dataset).sum(axis=1))[:, None]
    # Restore the original scale
    image_helper.ratio = parameters.ratio
    return features_queries, features_dataset

if __name__ == '__main__':

    parameters = Parameters()
    if not os.path.exists(parameters.temp_dir):
        os.makedirs(parameters.temp_dir)
    # np.array((104.00698793, 116.66876762, 122.67891434))
    # Load and reshape the means to subtract to the inputs
    parameters.means = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)[None, :, None, None]
    # parameters.means = np.array([103.93900299,  116.77899933,  123.68000031], dtype=np.float32)[None, :, None, None]

    # Configure caffe and load the network
    caffe.set_device(parameters.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(parameters.proto, parameters.weights, caffe.TEST)

    # Load the dataset and the image helper
    dataset = Dataset(parameters.dataset, parameters.eval_binary)
    image_helper = ImageHelper(parameters.ratio, parameters.L, parameters.means)

    # Extract features
    features_queries, features_dataset = extract_features(dataset, image_helper, net, parameters)

    # Database side expansion?
    if parameters.dbe is not None and parameters.dbe > 0:
        # Extend the database features
        # With larger datasets this has to be done in a batched way.
        # and using smarter ways than sorting to take the top k results.
        # For 5k images, not really a problem to do it by brute force
        X = features_dataset.dot(features_dataset.T)
        idx = np.argsort(X, axis=1)[:, ::-1]
        weights = np.hstack(([1], (parameters.dbe - np.arange(0, parameters.dbe)) / float(parameters.dbe)))
        weights_sum = weights.sum()
        features_dataset = np.vstack([np.dot(weights, features_dataset[idx[i, :parameters.dbe + 1], :]) / weights_sum for i in range(len(features_dataset))])

    # Compute similarity
    sim = features_queries.dot(features_dataset.T)
    # Average query expansion?
    if parameters.aqe is not None and parameters.aqe > 0:
        # Sort the results to get the nearest neighbors, compute average
        # representations, and query again.
        # No need to L2-normalize as we are on the query side, so it doesn't
        # affect the ranking
        idx = np.argsort(sim, axis=1)[:, ::-1]
        features_queries = np.vstack([np.vstack((features_queries[i], features_dataset[idx[i, :parameters.aqe]])).mean(axis=0) for i in range(len(features_queries))])
        #for i in range(features_queries.shape[0]):
        #    features_queries[i] = np.vstack((features_queries[i], features_dataset[idx[i, :parameters.aqe]])).mean(axis=0)
        sim = features_queries.dot(features_dataset.T)

    # Score
    dataset.score(sim, parameters.temp_dir, parameters.eval_binary)