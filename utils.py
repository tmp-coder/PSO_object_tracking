"""
-*- coding: utf-8 -*-
@Author  : zouzhitao
@Time    : 17-10-4 下午12:44
@File    : utils.py
"""

import numpy as np
from numpy.linalg import svd


def orth_u(u, e):
    """
    :param u: ndarray
    :param e: ndarray
    :return: orthonormal of [u | e]
    """
    assert u.shape[0] == e.shape[0]
    orthonormal = np.hstack((u, np.zeros(e.shape)))
    # Gram-Schmidt正交化
    for i in range(0, e.shape[1]):
        tmp = np.dot(e[:, i], orthonormal[:, 0:u.shape[1]+i])*orthonormal[:, 0:u.shape[1]+i]
        orthonormal[:, i+u.shape[1]] = e[:, i] - tmp.sum(1)
        orthonormal[:, i+u.shape[1]] = orthonormal[:, i+u.shape[1]]/np.linalg.norm(orthonormal[:, i+u.shape[1]], 2)
    return orthonormal


def rsvd(u, sigma, v, e):
    """
    r-svd algorithm
    A = u*sigma*v.T
    :param u: 2d_array
    :param sigma: diag matrix
    :param v: 2d_array
    :param e: 2d_array
    :return: svd of [A|e]
    """
    if len(sigma.shape) == 1:
        sigma = np.diag(sigma)
    m = e.shape[1]
    t = v.shape[1]
    u_wave = orth_u(u, e)
    e_wave = u_wave[:, u.shape[1]:]
    v_wave = np.hstack((np.vstack((v, np.zeros((m, t)))), np.vstack((np.zeros((t, m)), np.eye(m)))))
    assert v_wave.shape == (m+t, m+t)

    sigma_wave = np.hstack((np.vstack((sigma, np.zeros((e.shape[1], sigma.shape[1])))),
                            np.vstack((np.dot(u.T, e), np.dot(e_wave.T, e)))))
    u_hat, s_hat, v_hatH = svd(sigma_wave, full_matrices=False)
    # print(np.allclose(sigma_wave, np.dot(u_hat,np.dot(np.diag(s_hat),v_hatH))))
    return np.dot(u_wave,u_hat), s_hat, np.dot(v_hatH, v_wave.T)


def raw_svd(u, sigma, v, e):
    """
    A = u*sigma*v.T
    :param u: array(m,k)
    :param sigma: array(,k)
    :param v: array(n,k)
    :param e: array
    :return: svd of [A|e]
    """
    A_ = np.hstack((np.dot(u, np.dot(np.diag(sigma), v)), e))
    u_hat, s_hat, v_hatH = svd(A_)
    return u_hat, s_hat, v_hatH.T

if __name__ == '__main__':
    A = np.random.randn(4, 3)
    u, s, vh = svd(A, full_matrices=False)
    e = np.random.randn(4, 2)
    u_hat, s_hat, v_hat = rsvd(u, s, vh.T, e)
    tmp = np.dot(u_hat, np.diag(s_hat))
    tmp = np.dot(tmp, v_hat.T)
    A_ = np.hstack((A,e))
    print(np.allclose(A_, tmp))


