import numpy as np
import torch as T
import torch.nn as nn
from torch.autograd import Variable
import cv2

import util
import dotvis

# counting from 1:
# 0 ??
# 1 upperroot
# 2 rshoulder
# 3 relbow
# 4 rhand
# 5 lshoulder
# 6 lelbow
# 7 lhand
# 8 rhipjoint
# 9 rknee
# 10 rfoot
# 11 lhipjoint
# 12 lknee
# 13 lfoot
# 14 head
# 15 head
# 16 head
# 17 head

def run_network(model,
                image_path,
                padValue=128, stride=8,
                #scale_search=(0.5, 1., 1.5, 2.),
                scale_search=(0.5, 1.),
                boxsize=368):

    oriImg = cv2.imread(image_path) # B,G,R order
    imageShape = oriImg.shape

    imageToTest = Variable(T.transpose(T.transpose(T.unsqueeze(T.from_numpy(oriImg).float(), 0),2,3),1,2), volatile=True)

    multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]

    heatmap_avg = T.zeros((len(multiplier), 19, oriImg.shape[0], oriImg.shape[1]))
    paf_avg = T.zeros((len(multiplier), 38, oriImg.shape[0], oriImg.shape[1]))
    #print heatmap_avg.size()

    for m in range(len(multiplier)):
        scale = multiplier[m]
        #h = int(oriImg.shape[0]*scale)
        #w = int(oriImg.shape[1]*scale)
        #pad_h = 0 if (h % stride == 0) else stride - (h % stride)
        #pad_w = 0 if (w % stride == 0) else stride - (w % stride)
        #new_h = h+pad_h
        #new_w = w+pad_w

        imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
        imageToTest_padded = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5

        feed = Variable(T.from_numpy(imageToTest_padded))
        output1, output2 = model(feed)
        print 'paf', output1.size(), 'heatmap', output2.size()

        heatmap = nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1]))(output2)
        paf = nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1]))(output1)

        heatmap_avg[m] = heatmap[0].data
        paf_avg[m] = paf[0].data


    heatmap_avg = T.transpose(T.transpose(T.squeeze(T.mean(heatmap_avg, 0)),0,1),1,2)
    paf_avg     = T.transpose(T.transpose(T.squeeze(T.mean(paf_avg, 0)),0,1),1,2)
    heatmap_avg = heatmap_avg.cpu().numpy()
    paf_avg     = paf_avg.cpu().numpy()

    # heapmap_avg.shape == (h, w, nparts)
    # paf_avg.shape == (h, w, nparts*2)
    return heatmap_avg, paf_avg, imageShape


def peak_loc(hmap_ori, id_offset=0, thre1=0.1):
    """
    image -> [(peak_x, peak_y, image[peak_y, peak_x], id)]
    """
    from scipy.ndimage.filters import gaussian_filter
    hmap = gaussian_filter(hmap_ori, sigma=3)

    # maxpool-2d
    hmap_left = np.zeros(hmap.shape)
    hmap_left[1:,:] = hmap[:-1,:]
    hmap_right = np.zeros(hmap.shape)
    hmap_right[:-1,:] = hmap[1:,:]
    hmap_up = np.zeros(hmap.shape)
    hmap_up[:,1:] = hmap[:,:-1]
    hmap_down = np.zeros(hmap.shape)
    hmap_down[:,:-1] = hmap[:,1:]

    peaks_binary = np.logical_and.reduce((hmap >= hmap_left,
                                          hmap >= hmap_right,
                                          hmap >= hmap_up,
                                          hmap >= hmap_down,
                                          hmap > thre1))

    peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse

    return [(x, y, hmap_ori[y, x], id_offset + i) for i, (x,y) in enumerate(peaks)]


def get_peaks(heatmap_avg, thre1=0.1):
    all_peaks = []
    peak_counter = 0

    for part in range(18):
        peaks = peak_loc(heatmap_avg[:,:,part], id_offset=peak_counter)
        all_peaks.append(peaks)
        peak_counter += len(peaks)

    return all_peaks


# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10],
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17],
           [1,16], [16,18], [3,17], [6,18]]

# the middle joints heatmap correpondence
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22],
          [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52],
          [55,56], [37,38], [45,46]]


def score_connection(score_mid, peakA, peakB, imageShape, mid_num=10):
    vecA = peakA[:2]
    vecB = peakB[:2]

    scoreA = peakA[2]
    scoreB = peakB[2]

    jointvec = np.subtract(vecB, vecA)
    norm = np.linalg.norm(jointvec)
    vec = np.divide(jointvec, norm)

    zeros = np.zeros(mid_num, dtype=np.int)
    ones = np.ones(mid_num, dtype=np.int)
    xs = np.rint(np.linspace(vecA[0], vecB[0], num=mid_num)).astype(np.int)
    ys = np.rint(np.linspace(vecA[1], vecB[1], num=mid_num)).astype(np.int)

    vec_x = score_mid[ys, xs, zeros]
    vec_y = score_mid[ys, xs, ones]

    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
    score_with_dist_prior = score_midpts.mean() + min(0.5*imageShape[0]/norm-1, 0)

    return (score_midpts, score_with_dist_prior, score_with_dist_prior + scoreA + scoreB)


# joints - previously all_peaks
def get_connections2(imageShape, paf_avg, joints, mapIdx, limbSeq, thre2=0.05):
    connection_all = [] # for each viable limb: [(joint_index_A, joint_index_B, score, iA?, iB?)]
    lonely_limbs = []

    for k in range(len(mapIdx)): # for each viable limb
        score_mid = paf_avg[:,:,[x-19 for x in mapIdx[k]]]
        candA = joints[limbSeq[k][0]-1] # [(peak_x, peak_y, image[peak_y, peak_x], id)]
        candB = joints[limbSeq[k][1]-1]
        min_connections = min(len(candA), len(candB))

        connection_candidate = []
        for i, peakA in enumerate(candA):
            for j, peakB in enumerate(candB):
                score_midpts, score_with_dist_prior, score_sum = score_connection(score_mid, peakA, peakB, imageShape)
                criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                criterion2 = score_with_dist_prior > 0

                if criterion1 and criterion2:
                    connection_candidate.append([i, j, score_with_dist_prior, score_sum])

        if connection_candidate:
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if i not in connection[:,3] and j not in connection[:,4]:
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if len(connection) >= min_connections:
                        break

            connection_all.append(connection)
        else:
            lonely_limbs.append(k)
            connection_all.append([])

    return connection_all, lonely_limbs

def get_subset(all_peaks, connection_all, special_k, mapIdx, limbSeq):
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:,0]
            partBs = connection_all[k][:,1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])): #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    print "found = 2"
                    membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    return candidate, subset


WHITE = [254,1,1]

def visualize(imageShape, all_peaks, subset, candidate, limbSeq, colors):
    canvas = np.zeros(imageShape)
    for i in range(18):
        for j in range(len(all_peaks[i])):
            pt = all_peaks[i][j][0:2]
            cv2.circle(canvas, pt, 4, colors[i], thickness=-1)
            cv2.putText(canvas, str(i), pt,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        #color=colors[i],
                        color=WHITE,
                        thickness=1,
                        lineType=cv2.LINE_AA)

    stickwidth = 4

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i])-1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = np.degrees(np.arctan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas
