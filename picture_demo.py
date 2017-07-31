import network
import funs

# https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/4f0391dff042bfc13852989cbef9348bce5b739e/testing/src/connect56LineVec.m#L30

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10],
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17],
           [1,16], [16,18], [3,17], [6,18]]

# the middle joints heatmap correpondence
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22],
          [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52],
          [55,56], [37,38], [45,46]]

# visualize
colors = [[255, 0, 0],
          [255, 85, 0],
          [255, 170, 0],
          [255, 255, 0],
          [170, 255, 0],
          [85, 255, 0],
          [0, 255, 0],
          [0, 255, 85],
          [0, 255, 170],
          [0, 255, 255],
          [0, 170, 255],
          [0, 85, 255],
          [0, 0, 255],
          [85, 0, 255],
          [170, 0, 255],
          [255, 0, 255],
          [255, 0, 170],
          [255, 0, 85]]


def go(test_image='./sample_image/srb.png'):
    model = network.pose_model.create()

    heatmap_avg, paf_avg, imageShape = funs.run_network(model, test_image)
    all_peaks = funs.get_peaks(heatmap_avg)
    connection_all, special_k = funs.get_connections(imageShape, paf_avg, all_peaks, mapIdx, limbSeq, thre2=0.5)
    candidate, subset = funs.get_subset(all_peaks, connection_all, special_k, mapIdx, limbSeq)

    canvas = funs.visualize(imageShape, all_peaks, subset, candidate, limbSeq, colors)
    return canvas


if __name__ == '__main__':
    import cv2
    import sys

    args = sys.argv[1:3]
    if len(args) == 2:
        in_, out_ = args
    else:
        in_, out_ = './sample_image/srb.png', 'result.png'

    cv2.imwrite(out_, go(in_))
