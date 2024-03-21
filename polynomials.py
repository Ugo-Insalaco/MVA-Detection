import numpy as np
import tqdm

def poly_power(poly, n):
    deg = len(poly)-1
    if deg > 1: 
        raise NotImplementedError('degree is too large')
    if deg == 1:
        power = np.zeros(n+1)
        a, b = poly[0], poly[1]
        logs = np.log(np.arange(1, n+1))
        k = np.arange(0, n+1)
        nmk = n-k
        slk = np.cumsum(logs)
        slk = np.concatenate(([0], slk))
        slnmk = np.cumsum(np.flip(logs))
        slnmk = np.concatenate(([0], slnmk))
        l = k*np.log(b) + nmk * np.log(a) + slnmk - slk
        power = np.exp(l)
        return power
    if deg == 0:
        return [poly[0]**n]
    
def poly_mul_rec(poly1, poly2):
    d2 = len(poly2) - 1
    if d2 == 0:
        return poly2[0] * np.array(poly1)
    d1 = len(poly1) - 1
    deg = d1 + d2
    prod = np.zeros(deg+1)
    prod[:d1+1] = poly1
    recprod = poly_mul_rec(poly1, poly2[1:])
    recprod = np.concatenate(([0], recprod))
    return poly2[0] * prod + recprod

def poly_mul_iter(poly1, poly2):
    if len(poly1) < len(poly2):
        t = poly1
        poly1 = poly2
        poly2 = t
    d2 = len(poly2) - 1
    d1 = len(poly1) - 1
    deg = d1 + d2
    prod = np.zeros(deg+1)
    poly1 = np.array(poly1)
    poly2 = np.array(poly2)
    for i in range(d2+1):
        prod[i:i+d1+1] = prod[i:i+d1+1] + poly1 * poly2[i]
    return prod

def poly_list_mul(polys, verbose=False):
    n = len(polys)
    p = polys[0]
    iterator = tqdm.tqdm(range(1, n)) if verbose else range(1, n)
    for i in iterator:
        p = poly_mul_iter(p, polys[i])
    return p