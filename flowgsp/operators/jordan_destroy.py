"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
"""

from ..utils import *
from scipy import linalg

def find_best_pair(A:np.ndarray, k:np.ndarray, vl:np.ndarray, 
                   vr:np.ndarray, prefer_mask:Optional[np.ndarray], opt_eps:float=1e-5):
    
    """
    Find the best pairs of indices (i, j) in the matrix A that maximize the product of the corresponding left and right eigenvectors for the eigenvalue with index k.

    Parameters
    ----------
    A : np.ndarray
        The input matrix.
    k : np.ndarray
        The index of the eigenvalue to consider.
    vl : np.ndarray
        The left eigenvectors of A.
    vr : np.ndarray
        The right eigenvectors of A.
    opt_eps : float, optional
        The tolerance for selecting the best pairs, by default 1e-5.

    Returns
    -------
    np.ndarray
        The best pairs of indices (i, j) that maximize the product of the corresponding left and right eigenvectors.
    """

    mask = (A == 0).astype(float)
    cross_values = np.outer(np.abs(vl[:,k]), np.abs(vr[:,k]))
    if prefer_mask is not None:
        cross_values = cross_values * mask * prefer_mask
        best_pairs = np.asarray(np.where(np.abs(cross_values) > opt_eps)).T # Use different from 0 criterion
    else:
        cross_values = cross_values * mask
        best_pairs = np.asarray(np.where(np.abs(cross_values - cross_values.max()) < 1e-5)).T # Use maximum argument criterion

    return best_pairs

def destroy_jordan_blocks(A:np.ndarray, prefer_nodes:list=[], allow_self:bool=False, max_iter=10000):
    """
    Destroy the Jordan blocks in the input matrix A by setting the smallest entries to 1.

    Parameters
    ----------
    A : np.ndarray
        The input matrix (Adjacency matrix).
    prefer_nodes : list, optional
        A list of preferred node indices to prioritize when selecting the entries to set to 1, by default [].
    allow_self : bool, optional
        Whether to allow self-edges, by default False.
    max_iter : int, optional
        The maximum number of iterations to perform, by default 10000.

    Returns
    -------
    np.ndarray
        The modified matrix A with the Jordan blocks destroyed.
    """

    ret = deepcopy(A)
    _, vl, vr = linalg.eig(ret, left=True)
    n = len(A)
    cur_iter = 0
    space_tol = np.deg2rad(1)

    # Defining preferential masks for choice of edges
    mask_both = np.zeros_like(A)
    for p1 in prefer_nodes:
        for p2 in prefer_nodes:
            mask_both[p1,p2] = 1.0

    mask_one = np.zeros_like(A)
    for p1 in prefer_nodes:
        mask_one[p1, : ] = 1.0
        mask_one[:, p1] = 1.0

    while np.linalg.matrix_rank(vr) < n:
        D = np.nan_to_num(np.arccos(np.abs(vr.T@vr)))
        k = np.argmax(np.sum(D < space_tol, axis=1))

        both = find_best_pair(ret, k, vl, vr, mask_both)
        one = find_best_pair(ret, k, vl, vr, mask_one)
        none = find_best_pair(ret, k, vl, vr, None)

        if len(both) != 0: 
            for k in range(len(both)):
                i,j = both[k]
                if (i != j) or (allow_self): break
        elif len(one)!= 0: 
            for k in range(len(one)):
                i,j = one[k]
                if (i != j) or (allow_self): break
        else: 
            for k in range(len(none)):
                i,j = none[k]
                if (i != j) or (allow_self): break

        if (i == j and (not allow_self)):
            continue
        ret[i,j] = 1
        _, vl ,vr = linalg.eig(ret, left=True)


        if cur_iter > max_iter:
            break
        cur_iter += 1        

    return ret

def destroy_jordan_blocks_laplacian(A:np.ndarray, prefer_nodes:list=[], max_iter=10000):
    """
    Destroy the Jordan blocks in the input matrix A by setting the smallest entries to 1.

    Parameters
    ----------
    A : np.ndarray
        The input matrix (Laplacian matrix).
    prefer_nodes : list, optional
        A list of preferred node indices to prioritize when selecting the entries to set to 1, by default [].
    allow_self : bool, optional
        Whether to allow self-edges, by default False.
    max_iter : int, optional
        The maximum number of iterations to perform, by default 10000.

    Returns
    -------
    np.ndarray
        The modified matrix A with the Jordan blocks destroyed.
    """

    ret = deepcopy(A)
    _, vl, vr = linalg.eig(ret, left=True)
    n = len(A)
    cur_iter = 0
    space_tol = np.deg2rad(1)

    # Defining preferential masks for choice of edges
    mask_both = np.zeros_like(A)
    for p1 in prefer_nodes:
        for p2 in prefer_nodes:
            mask_both[p1,p2] = 1.0

    mask_one = np.zeros_like(A)
    for p1 in prefer_nodes:
        mask_one[p1, : ] = 1.0
        mask_one[:, p1] = 1.0

    while np.linalg.matrix_rank(vr) < n:
        D = np.nan_to_num(np.arccos(np.abs(vr.T@vr)))
        k = np.argmax(np.sum(D < space_tol, axis=1))

        both = find_best_pair(ret, k, vl, vr, mask_both)
        one = find_best_pair(ret, k, vl, vr, mask_one)
        none = find_best_pair(ret, k, vl, vr, None)

        if len(both) != 0: 
            for k in range(len(both)):
                i,j = both[k]
        elif len(one)!= 0: 
            for k in range(len(one)):
                i,j = one[k]
        else: 
            for k in range(len(none)):
                i,j = none[k]

        # Laplacian constraints -> updating diagonal
        ret[i,j] = -1
        ret[i,i] += 1
        _, vl ,vr = linalg.eig(ret, left=True)

        if cur_iter > max_iter:
            break
        cur_iter += 1        

    return ret


def destroy_zero_eigenvals(A:np.ndarray, prefer_nodes:list=[], eps:float=1e-6, tol:float=1e-4, 
                           allow_self:bool=False, max_iter=10000, verbose:bool=False):
    """
    Destroy the zero eigenvalues in the input matrix A by setting the smallest entries to 1.

    Parameters
    ----------
    A : np.ndarray
        The input matrix.
    prefer_nodes : list, optional
        A list of preferred node indices to prioritize when selecting the entries to set to 1, by default [].
    eps : float, optional
        The tolerance value for considering an eigenvalue as zero, by default 1e-6.
    tol : float, optional
        The tolerance value for selecting the best pairs, by default 1e-4.
    allow_self : bool, optional
        Whether to allow self-edges, by default False.
    max_iter : int, optional
        The maximum number of iterations to perform, by default 10000.
    verbose : bool, optional
        Whether to print the dimension of the null space at each iteration, by default

    Returns
    -------
    np.ndarray
        The modified matrix A with the zero eigenvalues destroyed.
    """
    
    ret = deepcopy(A)
    D, vl, vr = linalg.eig(ret, left=True)
    repeat_count = 0
    prev_rank = -1
    cur_iter = 0

    # Defining preferential masks for choice of edges
    mask_both = np.zeros_like(A)
    for p1 in prefer_nodes:
        for p2 in prefer_nodes:
            mask_both[p1,p2] = 1.0

    mask_one = np.zeros_like(A)
    for p1 in prefer_nodes:
        mask_one[p1, : ] = 1.0
        mask_one[:, p1] = 1.0

    while np.min(np.abs(D)) < eps:

        possible_index = np.where((np.abs(D) - np.min(np.abs(D))) < tol)[0]
        cur_rank = len(possible_index)
        if verbose:
            print(f"Dimension of null space={cur_rank}")

        if cur_rank == prev_rank:
            repeat_count += 1
        else:
            repeat_count = 0
        if repeat_count >= 10:
            if verbose: print("Remove preferential nodes")
            both, one = [], []
            none = find_best_pair(ret, np.argmin(np.abs(D)), vl, vr, None)
        else:
            both = [find_best_pair(ret, possible_index[k], vl, vr, mask_both) for k in range(len(possible_index))]
            both = np.concatenate(both)
            one = [find_best_pair(ret, possible_index[k], vl, vr, mask_one) for k in range(len(possible_index))]
            one = np.concatenate(one)
            none = [find_best_pair(ret, possible_index[k], vl, vr, None) for k in range(len(possible_index))]
            none = np.concatenate(none)


        if len(both) != 0: 
            for k in range(len(both)):
                i,j = both[k]
                if (i != j) or (allow_self): break
        elif len(one)!= 0: 
            for k in range(len(one)):
                i,j = one[k]
                if (i != j) or (allow_self): break
        else: 
            for k in range(len(none)):
                i,j = none[k]
                if (i != j) or (allow_self): break

        if (i == j and (not allow_self)):
            continue
        ret[i,j] = 1

        # Update
        D, vl ,vr = linalg.eig(ret, left=True)
        prev_rank = cur_rank

        if cur_iter > max_iter:
            break
        cur_iter += 1


    return ret