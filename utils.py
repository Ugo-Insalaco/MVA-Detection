from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def subsample(img):
    s= 2
    return img[::s, ::s]

def load_grey(path):
    im = Image.open(path)
    im = np.mean(np.array(im), axis = 2)
    return im

def add_marker(img, h, w, r, alpha):
    thickness = np.floor(min(img.shape[0], img.shape[1])/200+1).astype(np.int32)
    m = int(np.max(img))
    if m == 0: 
        m = 1
    img_circle = cv2.circle(img.copy(), (w, h), r, (m, m, m), thickness=thickness)
    img_circle = cv2.circle(img_circle, (w, h), int(alpha*r), (m, m, m), thickness=thickness)
    return img_circle

def add_circle(img, h, w, r):
    m = int(np.max(img))
    if m == 0: 
        m = 1
    img_circle = cv2.circle(img.copy(), (w, h), r, (m, m, m), thickness=5)
    return img_circle

def center_clip_marker_array(c, r, arr, plot = False): 
    # arr: H x W x ...
    H, W = arr.shape[0], arr.shape[1]
    rf = np.floor(r).astype(np.int32)
    redmy, redMy = max(0, c[0] - rf), min(H, c[0] + rf)
    redmx, redMx = max(0, c[1] - rf), min(W, c[1] + rf)
    reduced_arr = arr[redmy: redMy, redmx: redMx]
    if plot: 
        ls = len(arr.shape)
        if ls > 2 or (ls == 3 and arr.shape[2] != 3):
            raise ValueError('Cannot plot an array with more than 2 dimensions or invalid color channel')
        plt.figure()
        plt.imshow(reduced_arr, cmap='grey')
    return reduced_arr

def center_clip_marker_list(c, r, lst):
    # lst: Q x 2
    rf = np.floor(r).astype(np.int32)
    redmy = max(0, c[0] - rf)
    redmx = max(0, c[1] - rf)
    reduced_lst = lst - np.array([[redmy, redmx]]) - 1
    return reduced_lst

def save_detection(zrcs, hfound, wfound, rfoundi, hpruned, wpruned, rprunedi, lambd, rs, alpha, H, W, file_name):
    detection = np.stack((hfound, wfound, rfoundi), axis = 1)
    pruned = np.stack((hpruned, wpruned, rprunedi), axis = 1)
    np.savez(file_name, zrcs=zrcs, detection=detection, pruned=pruned, lambd=lambd, rs=rs, alpha=alpha, H=H, W=W)

def test_parameters(base_img, alpha, c, rmin, rmax, nr):
    img = subsample(base_img)    
    rs = np.linspace(rmin, rmax, nr)
    fig_per_line = 3
    if nr//fig_per_line == 0:
        _, axs = plt.subplots(1, nr)
        if nr == 1:
            axs = np.array([[axs]])
        else:
            axs = axs[None, :]
    else:
        _, axs = plt.subplots(nr//fig_per_line, fig_per_line)
    for i in range(nr):
        r = np.floor(rs[i]).astype(np.int64)
        img_circle = add_marker(img, c[0], c[1], r, alpha)
        
        axs[i//fig_per_line, i%fig_per_line].imshow(img_circle, cmap='grey')
        axs[i//fig_per_line, i%fig_per_line].set_title(fr'r={r}, $\alpha=${alpha}')
        axs[i//fig_per_line, i%fig_per_line].set_axis_off()