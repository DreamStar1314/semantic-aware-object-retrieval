# -*- coding: utf-8 -*-
import numpy as np

def vecpostproc(x,a=1):
    #print x.shape,(sum(x**2)).shape # 512*20, 20
    if a!=1:
        x = np.sign(a)*(abs(x) **a)
    x = np.nan_to_num(x/np.sqrt(sum(x**2)))
    where_are_inf=np.isinf(x)
    x[where_are_inf]= 0
    #x = np.where(abs(x) < 1e5, x, 0)
    return x.transpose()

def pack_regions_for_network(seg_map,all_regions):
    n_regs = np.sum([len(e) for e in all_regions]) #一个特征图提出三种level的特征特征框的总数
    # print all_regions[0]
    R = np.zeros((n_regs, 5), dtype=np.float32)
    cnt = 0
    # There should be a check of overflow...
    for i, r in enumerate(all_regions):
        try:
            R[cnt:cnt + r.shape[0], 0] = i
            R[cnt:cnt + r.shape[0], 1:] = r
            cnt += r.shape[0]
        except:
            continue
    assert cnt == n_regs
    R = R[:n_regs] #R(:,0)=0,R(:n_regs,1:) = r
    # regs where in xywh format. R is in xyxy format, where the last coordinate is included. Therefore...
    R[:n_regs, 3] = R[:n_regs, 1] + R[:n_regs, 3] - 1 # R[:n_regs, 1]左边起始点，R[:n_regs, 3]右边所在点
    R[:n_regs, 4] = R[:n_regs, 2] + R[:n_regs, 4] - 1 # R[:n_regs, 2]上边起始点，R[:n_regs, 4]下边所在点

    return R

def get_rmac_region_coordinates(H,W,L):

    ovr = 0.4
    # Possible regions for the long dimension
    steps = np.array((2, 3, 4, 5, 6, 7), dtype=np.float32)
    w = np.minimum(H, W)

    b = (np.maximum(H, W) - w) / (steps - 1) #将H-W分别等分为1,2,...,6份时，每份长度为b[i]
    # steps(idx) regions for long dimension. The +1 comes from Matlab 1-indexing...
    # (w^2 - w * b) / w ^2- ovr(0.4) = 0.6-w/b，即对H-W划出更接近重合率0.4的等份，存在不足或过剩
    # **************
    idx = np.argmin(np.abs(((w ** 2 - w * b) / w ** 2) - ovr)) + 1 #idx表示沿着最长边，要分割多少份

    #*************
    # Region overplus per dimension
    Wd = 0
    Hd = 0
    if H < W:
        Wd = idx
    elif H > W:
        Hd = idx

    regions_xywh = []
    for l in range(1, L + 1): # 【1,2,...,L】
        wl = np.floor(2 * w / (l + 1)) #RMAC的region的宽和高,np.floor返回浮点数， 每个level生成 l * (l + m -1) 个宽为 2 min(W,H)/(l + 1)的region
        wl2 = np.floor(wl / 2 - 1) #第一个region的一个中心坐标，有小数位
        # Center coordinates
        if l + Wd - 1 > 0: #认为 H>W , Wd = 0 ;
            b = (W - wl) / (l + Wd - 1) #沿着边W，第一个region后的了l-1个region的相邻右边的间隔或平分为l-1份的每份长度
        else:
            b = 0
        cenW = np.floor(wl2 + b * np.arange(l - 1 + Wd + 1)) - wl2 #cenW列表每个元素,表示沿着边W,每个框的近似左边起始点, l=1,b=0,cenW=0
        # Center coordinates
        if l + Hd - 1 > 0:
            b = (H - wl) / (l + Hd - 1)
        else:
            b = 0
        cenH = np.floor(wl2 + b * np.arange(l - 1 + Hd + 1)) - wl2 #cenH列表每个元素,表示沿着边H,每个框的近似上边起始点

        for i_ in cenH:
            for j_ in cenW:
                regions_xywh.append([j_, i_, wl, wl])

    # print "regions number: %d" % len(regions_xywh)
    # Round the regions. Careful with the borders!
    for i in range(len(regions_xywh)):
        for j in range(4):
            regions_xywh[i][j] = int(round(regions_xywh[i][j]))
        if regions_xywh[i][0] + regions_xywh[i][2] > W:
            regions_xywh[i][0] -= ((regions_xywh[i][0] + regions_xywh[i][2]) - W)
        if regions_xywh[i][1] + regions_xywh[i][3] > H:
            regions_xywh[i][1] -= ((regions_xywh[i][1] + regions_xywh[i][3]) - H)
    return np.array(regions_xywh).astype(np.float32)


def get_rmac_features(X, R):

    nfeat = []
    L2 = []
    L2_FV = []
    L2_FV_Reduce = {}

    for r in R:
        if (r[2] != r[4]) and (r[1] != r[3]):
            newX = X[:, int(r[2]):int(r[4]), int(r[1]):int(r[3])]
            newX = np.nan_to_num(newX)
            feat = newX.max(1).max(-1) #每个区域的极大值
            #nfeat.append(feat)
            L2.append(feat.sum())
            L2_FV.append(feat)

    #############################
    L2_ = np.argsort(np.array(L2))
    for i in range(int(np.floor(len(L2_)*1.0))):
        nfeat.append(L2_FV[i])
    #########################################

    if len(nfeat):
        nfeat = np.array(nfeat)
        nfeat = [element for element in vecpostproc(nfeat.transpose())] # vecpostproc imporve the map= 86.1 to 87.53
    else:
        nfeat = [np.zeros(X.shape[0])]  #jesus_000280.jpg\oxford_001928.jpg
    return nfeat # 512*--


def apply_rmac_aggregation(X,seg_map=[]):

    L = 3
    k, h, w = X.shape
    all_regions = []
    all_regions.append(get_rmac_region_coordinates(h, w, L))#获取所有中心点坐标和作用区域的w和h；
    R = pack_regions_for_network(seg_map,all_regions) # R（，0）表示all_regions的元素的index, R（，1:4）四个元素表示框的区域
    feat = get_rmac_features(X, R) #返回的是标准化后的全局极大、各个level的region的极大；
    return feat #20*512




