cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef extern from "math.h":
    double abs(double m)
    double log(double x)




def build_tg(   np.ndarray[DTYPE_t,ndim=2] gtclses,
                np.ndarray[DTYPE_t,ndim=2] gtboxes,
                np.ndarray[DTYPE_t,ndim=2] ious,
                np.ndarray[unsigned char,ndim=2] pos_mask,
                np.ndarray[DTYPE_t,ndim=2] DefBoxes
                ):
    return build_tg_c(gtclses,gtboxes,ious,pos_mask,DefBoxes)

cdef np.ndarray[DTYPE_t, ndim=2] build_tg_c(np.ndarray[DTYPE_t,ndim=2] gtclses,
                np.ndarray[DTYPE_t,ndim=2] gtboxes,
                np.ndarray[DTYPE_t,ndim=2] ious,
                np.ndarray[unsigned char,ndim=2] pos_mask,
                np.ndarray[DTYPE_t,ndim=2] DefBoxes):
    cdef unsigned int N = ious.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] tgboxes = np.zeros((8732,4),dtype = DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] tgcls = np.zeros((8732,21),dtype = DTYPE)
    cdef DTYPE_t xmin,ymin,xmax,ymax,w,h,x,y,cls

    for i in range(N):
        xmin, ymin, xmax, ymax = gtboxes[i][0], gtboxes[i][1], gtboxes[i][2], gtboxes[i][3]
        w = xmax - xmin
        h = ymax - ymin
        x = xmin + 0.5 * w
        y = ymin + 0.5 * h
        cls = gtclses[i]
        indice = np.nonzero(pos_mask[i])
        if(len(indice)!=0):
   
            for ind in indice[0]:


                defboxes = DefBoxes[ind,:]
                tgboxes[ind][0]= (x-defboxes[0])/defboxes[2]
                tgboxes[ind][1]= (y-defboxes[1])/defboxes[3]
                tgboxes[ind][2]= np.log(w/defboxes[2])
                tgboxes[ind][3]= np.log(h/defboxes[3])
                tgcls[ind][int(cls)]=1
        else:
            pass
    return np.concatenate((tgboxes,tgcls),1)

def bbox_ious(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):
    return bbox_ious_c(boxes, query_boxes)


cdef np.ndarray[DTYPE_t, ndim=2] bbox_ious_c(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    For each query box compute the IOU covered by boxes
    ----------
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of intersec between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] intersec = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, qbox_area, box_area, inter_area
    cdef unsigned int k, n
    for k in range(K):
        qbox_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] ) *
            (query_boxes[k, 3] - query_boxes[k, 1] )
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0])
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1])
                )
                if ih > 0:
                    box_area = (
                        (boxes[n, 2] - boxes[n, 0] ) *
                        (boxes[n, 3] - boxes[n, 1] )
                    )
                    inter_area = iw * ih
                    intersec[n, k] = inter_area / (qbox_area + box_area - inter_area)
    return intersec




