import numpy as np
import matplotlib.pyplot as plt
from fast_histogram import histogram1d
from polynomials import poly_list_mul, poly_power
from tqdm.contrib.itertools import product
from utils import add_marker, add_circle, center_clip_marker_array, center_clip_marker_list, subsample, save_detection
from gradients import compute_gradients
import tqdm

def compute_pkcqr(c, r, alpha, q, use_norm=False):
    # c: 2
    # q: N x 2
    norm = np.linalg.norm((q-c[None, :]), axis = 1) if not use_norm else q
    return 2/np.pi*np.arctan(alpha * r/norm) # N


def get_points_in_crown(c, r, alpha, H, W, return_norms = False, plot = False):
    # c: 2
    rf = np.floor(r).astype(np.int32)
    redmy, redMy = max(0, c[0] - rf), min(H, c[0] + rf)
    redmx, redMx = max(0, c[1] - rf), min(W, c[1] + rf)
    YX = np.mgrid[redmy:redMy,redmx:redMx]
    YXc = YX - c[:, None, None]
    norm = np.linalg.norm(YXc, axis=0) # H x W
    cond = (norm > alpha*r)&(norm < r) # H x W
    Ycond, Xcond  = np.nonzero(cond)
    if return_norms:
        outnorms = norm[Ycond, Xcond]
    YXcond = np.stack((Ycond, Xcond), axis = 1)
    YXcond = YXcond + np.array([[redmy, redmx]])

    if plot:
        im = np.zeros((H, W))
        im[YXcond[:, 0], YXcond[:, 1]] = 1
        plt.figure()
        plt.imshow(im)

    if return_norms:
        return YXcond, outnorms
    else: 
        return YXcond

def get_points_in_crown_exclude_crown(c1, r1, c2, r2, alpha, H, W, return_norms = False, plot = False):
    # c: 2
    rf1 = np.floor(r1).astype(np.int32)
    rf2 = np.floor(r2).astype(np.int32)
    redmy1, redMy1 = max(0, c1[0] - rf1), min(H, c1[0] + rf1)
    redmx1, redMx1 = max(0, c1[1] - rf1), min(W, c1[1] + rf1)
    redmy2, redMy2 = max(0, c2[0] - rf2), min(H, c2[0] + rf2)
    redmx2, redMx2 = max(0, c2[1] - rf2), min(W, c2[1] + rf2)
    redmy, redMy = min(redmy1, redmy2), max(redMy1, redMy2)
    redmx, redMx = min(redmx1, redmx2), max(redMx1, redMx2)

    YX = np.mgrid[redmy:redMy,redmx:redMx]
    YXc1 = YX - c1[:, None, None]
    YXc2 = YX - c2[:, None, None]
    norm1 = np.linalg.norm(YXc1, axis=0) # R x R
    norm2 = np.linalg.norm(YXc2, axis=0) # R x R
    cond1 = (norm1 > alpha*r1)&(norm1 < r1)
    cond2 = (1-((norm2 > alpha*r2)&(norm2 < r2)))
    cond = cond1&cond2 # R x R
    Ycond, Xcond  = np.nonzero(cond)
    if return_norms:
        outnorms = norm1[Ycond, Xcond]
    YXcond = np.stack((Ycond, Xcond), axis = 1)
    YXcond = YXcond + np.array([[redmy, redmx]])

    if plot:
        im = np.zeros((H, W))
        im[YXcond[:, 0], YXcond[:, 1]] = 1
        plt.figure()
        plt.imshow(im, cmap='grey')

    if return_norms:
        return YXcond, outnorms
    else: 
        return YXcond
    
def get_points_in_circle(c, r, H, W, return_norms = False, plot = False):
    # c: 2
    rf = np.floor(r).astype(np.int32)
    redmy, redMy = max(0, c[0] - rf), min(H, c[0] + rf)
    redmx, redMx = max(0, c[1] - rf), min(W, c[1] + rf)
    YX = np.mgrid[redmy:redMy,redmx:redMx]
    YXc = YX - c[:, None, None]
    norm = np.linalg.norm(YXc, axis=0) # H x W
    cond = (norm < r) # H x W
    Ycond, Xcond  = np.nonzero(cond)
    if return_norms:
        outnorms = norm[Ycond, Xcond]
    YXcond = np.stack((Ycond, Xcond), axis = 1)
    YXcond = YXcond + np.array([[redmy, redmx]])

    if plot:
        im = np.zeros((H, W))
        im[YXcond[:, 0], YXcond[:, 1]] = 1
        plt.figure()
        plt.imshow(im)
            
    if return_norms:
        return YXcond, outnorms
    else: 
        return YXcond
     
def compute_generating(pkcqr, verbose=False):
    N = pkcqr.shape[0]
    B = pkcqr
    A = 1-pkcqr
    p = np.zeros(N+1)
    p[0] = A[0]
    p[1] = B[0]
    for i in range(1, N):
        p[i+1] = (p[i]*B[i]).copy()
        p[1:i+1] = (p[1:i+1]*A[i] + p[0:i]*B[i]).copy()
        p[0] = (p[0]*A[i]).copy()
    if verbose:
        print('Proba sum (should be 1):', np.sum(p))
    return p

def get_crown_hist(c, r, alpha, H, W, b, plot=False):
    _, norm = get_points_in_crown(c, r, alpha, H, W, return_norms=True)
    hist = histogram1d(norm, range = [alpha*r, r], bins=b)
    hist = hist.astype(np.int32)
    x = np.linspace(alpha*r, r, b)
    if plot: 
        plt.figure()
        plt.plot(x, hist)
        plt.figure()
        plt.hist(norm, bins = b)
    return x, hist

def compute_generating_hist(c, r, alpha, norm, hist, verbose=False):
    # norm: b
    # hist: b
    b = len(hist)
    pkcqr = compute_pkcqr(c, r, alpha, norm, use_norm=True)
    raw_polys = np.stack((1-pkcqr, pkcqr), axis = 1) # b x 2
    if verbose:
        power_polys = []
        print('Computing poly powers')
        for i in tqdm.tqdm(range(b)):
            power_polys.append(poly_power(raw_polys[i], hist[i]))
    else:
        power_polys = [poly_power(raw_polys[i], hist[i]) for i in range(b)]
    if verbose:
        print('Computing poly multiplication')
    gen = poly_list_mul(power_polys, verbose=verbose)
    return gen

def get_lambda_r(r, alpha, H, W, b, epsm):
    c = np.array([int(H/2), int(W/2)])
    norm, hist = get_crown_hist(c, r, alpha, H, W, b)
    gen = compute_generating_hist(c, r, alpha, norm, hist)
    n = gen.shape[0]
    cs = np.cumsum(np.flip(gen))
    return n-np.argmax(cs >= epsm)

def get_lambdas(rs, alpha, H, W, b, eps):
    lambd = []
    nr = len(rs)
    epsm = eps/(nr * H * W)
    for r in tqdm.tqdm(rs):
        lambd.append(get_lambda_r(r, alpha, H, W, b, epsm))
    lambd = np.array(lambd)
    return lambd

def get_meaningfulness(r, alpha, H, W, b, zcr, verbose=False):
    c = np.array([int(H/2), int(W/2)])
    norm, hist = get_crown_hist(c, r, alpha, H, W, b)
    gen = compute_generating_hist(c, r, alpha, norm, hist, verbose=verbose)
    n = gen.shape[0]
    cs = np.cumsum(np.flip(gen))
    return cs[n-zcr]

def compute_zcr(c, ralpha, q, orths, plot = False):
    # orths: (H x W)x2
    # q: Q x 2, points on which to compute the zcr
    qorths = orths[q[:, 0], q[:, 1]]
    
    # computing dot products (cos)
    qc = (c[None, :] - q) # Q x 2
    norms = np.linalg.norm(qc, keepdims=1, axis = 1)
    qc = qc/norms
    dots = np.sum(qorths * qc, axis = 1) # Q
    s = np.sign(dots)
    qorths = s[...,None] * qorths
    dots = s * dots

    # computing dets (sin)
    pre_dets = np.stack((qorths, qc), axis = 2) # Q x 2 x 2
    dets = pre_dets[:, 0, 0] * pre_dets[:, 1, 1] - pre_dets[:, 1, 0] * pre_dets[:, 0, 1]

    # computing detections
    dets[dots == 0] = np.inf
    dots[dots == 0] = 1
    tans = dets/dots
    detections = (tans > -ralpha/norms[:,0])&(tans < ralpha/norms[:, 0])

    if plot: 
        new_img = orths[:, :, 1]
        new_img = (new_img - new_img.min())/(new_img.max() - new_img.min())*255
        new_img = new_img.astype(np.uint8)
        new_img = np.repeat(new_img[..., None], 3, axis = 2)
        new_img = add_circle(new_img, c[0], c[1], np.floor(ralpha).astype(np.int64))
        qdetected = q[detections]
        new_img[qdetected[:, 0], qdetected[:, 1]] = np.array([255, 0, 0])
        plt.figure()
        plt.imshow(new_img, cmap='grey')

    return np.sum(detections)

def plot_zcr(base_img, c, r, alpha):
    grads = compute_gradients(base_img)
    gradsx, gradsy = grads
    orths = np.stack([-gradsx, gradsy], axis = 2)
    orths = orths/np.linalg.norm(orths, axis = 1, keepdims=1)

    img = subsample(base_img)
    H, W = img.shape[0], img.shape[1]

    reduced_orths = center_clip_marker_array(c, r, orths)
    q = get_points_in_crown(c, r, alpha, H, W, return_norms=False)
    reduced_q = center_clip_marker_list(c, r, q)
    reduced_c = center_clip_marker_list(c, r, c[None,:])[0]
    zcr = compute_zcr(reduced_c, r*alpha, reduced_q, reduced_orths, plot=True)
    print(f'found {zcr} well oriented pixels over {len(q)} ({round(zcr/len(q)*100, 2)}%)')

def detect(rs, alpha, grads):
    gradsx, gradsy = grads
    orths = np.stack([-gradsx, gradsy], axis = 2)
    orths = orths/np.linalg.norm(orths, axis = 1, keepdims=1)
    H, W = orths.shape[0], orths.shape[1]
    nr = len(rs)

    zrcs = np.zeros((H, W, nr))
    for h, w in product(range(H), range(W)):
        c = np.array([h, w])
        for ri in range(len(rs)):
            r = rs[ri]
            reduced_orths = center_clip_marker_array(c, r, orths)
            q = get_points_in_crown(c, r, alpha, H, W, return_norms=False)
            reduced_q = center_clip_marker_list(c, r, q)
            reduced_c = center_clip_marker_list(c, r, c[None,:])[0]
            zrc = compute_zcr(reduced_c, r*alpha, reduced_q, reduced_orths)
            zrcs[h, w, ri] = zrc
    return zrcs

def meaningful_detection(zrcs, lambd, sort=True):
    zsl = zrcs/lambd[None, None, :] # H x W x 2
    detections = zsl > 1
    hfound, wfound, rfoundi = np.nonzero(detections) # Q, Q, Q

    if sort:
        zsldetected = zsl[hfound, wfound, rfoundi] # Q
        sorti = np.flip(np.argsort(zsldetected)) # Q, sorted in desc order
        hfound = hfound[sorti]
        wfound = wfound[sorti]
        rfoundi = rfoundi[sorti]
    return hfound, wfound, rfoundi

def aggregate_detections(hfound, wfound, rfoundalpha, H, W, plot = False):
    # hfound, wfound, rfound: N
    mask = np.zeros((H, W)) # H x W
    N = len(hfound)
    for i in tqdm.tqdm(range(N)):
        c = np.array([hfound[i], wfound[i]])
        in_circle = get_points_in_circle(c, rfoundalpha[i], H, W)
        mask[in_circle[:, 0], in_circle[:, 1]] = 1
    if plot:
        plt.figure()
        plt.imshow(mask, cmap='grey')
    return mask

def masked_detection(c1, r1, c2, r2, alpha, orths, lambd, verbose=False):
    # Is detection of 1 masked by 2 ?
    if np.linalg.norm(c2-c1) > r1+r2:
        if verbose:
            print('circles are too far appart')
        return False # Crowns are too far appart to be masking
    H, W = orths.shape[0], orths.shape[1]
    q = get_points_in_crown_exclude_crown(c1, r1, c2, r2, alpha, H, W, plot=verbose)
    if q.shape[0] < lambd:
        if verbose:
            print('too many points removed')
        return True # We have removed too many points to still be meaningful
    reduced_orths = center_clip_marker_array(c1, r1, orths)
    reduced_q = center_clip_marker_list(c1, r1, q)
    reduced_c = center_clip_marker_list(c1, r1, c1[None,:])[0]
    zrc = compute_zcr(reduced_c, r1*alpha, reduced_q, reduced_orths)
    if verbose:
        print('foud remaining:', zrc, 'points', lambd, 'to be meaningful')
    return zrc < lambd

def prune_detection(rs, alpha, orths, zrcs, lambd):
    # zrcs: H x W x nr
    # lambd: nr
    # orths: H x W x 2 
    hfound, wfound, rfoundi = meaningful_detection(zrcs, lambd)
    rfound = rs[rfoundi]
    
    N = hfound.shape[0]
    kept_detections = []
    for i in tqdm.tqdm(range(N)):
        c1 = np.array([hfound[i], wfound[i]])
        r1 = rfound[i]
        masked = False
        for k in kept_detections:
            c2 = np.array([hfound[k], wfound[k]])
            r2 = rfound[k]
            if masked_detection(c1, r1, c2, r2, alpha, orths, lambd[rfoundi[i]]):
                masked = True
                break
        if masked == False:
            kept_detections.append(i)
    kept_detections = np.array(kept_detections)
    if kept_detections.shape[0] == 0:
        hpruned, wpruned, rprunedi = np.array([]), np.array([]), np.array([])
    else:
        hpruned, wpruned, rprunedi = hfound[kept_detections], wfound[kept_detections], rfoundi[kept_detections]
    return hpruned, wpruned, rprunedi


def plot_detections(img, alpha, hfound, wfound, rfound, detectmin=None, detectmax=None):
    _, axs = plt.subplots(1, 3, figsize=(10,6))
    for ax in axs:
        ax.set_axis_off()
    axs[0].imshow(img, cmap='grey')
    axs[0].set_title('(a) Base image')

    H, W = img.shape[0], img.shape[1]
    new_img = img.copy()
    if detectmin == None:
        detectmin = 0
    if detectmax == None: 
        detectmax = hfound.shape[0]

    detectmin = min(max(detectmin, 0), hfound.shape[0])
    detectmax = min(max(detectmax, detectmin), hfound.shape[0])
    for i in range(detectmin, detectmax):
        new_img = add_marker(new_img, hfound[i], wfound[i], rfound[i], alpha)
        # new_img[hfound[i], wfound[i]] = 255

    axs[1].imshow(new_img, cmap='grey')
    axs[1].set_title('(b) Detections')
    
    print("computing aggregated map")
    detection_mask = aggregate_detections(hfound, wfound, rfound*alpha, H, W)  
    axs[2].imshow(detection_mask, cmap='grey')
    axs[2].set_title('(c) Detection mask')

def full_detection(base_img, rmin, rmax, nr, alpha, eps,b, save_file):
    grads = compute_gradients(base_img)
    gradsx, gradsy = grads
    
    orths = np.stack([-gradsx, gradsy], axis = 2)
    orths = orths/np.linalg.norm(orths, axis = 1, keepdims=1)
    
    img = subsample(base_img)
    H, W = img.shape[0], img.shape[0]
    
    rs = np.floor(np.linspace(rmin, rmax, nr)).astype(np.int64)
    print("=== 1/5 Computing lambda values ===")
    lambd = get_lambdas(rs, alpha, H, W, b, eps)

    print("=== 2/5 Computing Zrc values === ")
    zrcs = detect(rs, alpha, grads)

    max_detect_show = 5000
    print("=== 3/5 Filtering meaningful detections ===")
    hfound, wfound, rfoundi = meaningful_detection(zrcs, lambd)
    rfound = rs[rfoundi]
    plot_detections(img, alpha, hfound, wfound, rfound, 0, max_detect_show)
    print(f"found {len(hfound)} detections")
    
    print("=== 4/5 Pruning detections with masking principle ===")
    hpruned, wpruned, rprunedi = prune_detection(rs, alpha, orths, zrcs, lambd)
    rpruned = rs[rprunedi]    
    plot_detections(img, alpha, hpruned, wpruned, rpruned)

    print(f"=== 5/5 Saving results to {save_file} ===")
    save_detection(zrcs, hfound, wfound, rfoundi, hpruned, wpruned, rprunedi, lambd, rs, alpha, H, W, save_file)