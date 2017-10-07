"""
-*- coding: utf-8 -*-
@Author  : zouzhitao
@Time    : 17-10-4 下午12:44
@File    : utils.py
"""

import numpy as np
from numpy.linalg import svd


def bound(shape, left, right):
    return  left[0]>=0 and left[1]>=0 and left[0]<= shape[0] and left[1]<= shape[1]\
            and right[1] <= shape[1] and right[0] <= shape[0]

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


class State(object):
    # 形框最大大小
    MAX_LEN = 90

    def __init__(self, l, r):
        xmin, ymin = l
        xmax, ymax = r
        self.xmin = xmin
        self.ymin = ymin
        w = xmax - xmin
        h = ymax - ymin
        if w > self.MAX_LEN:
            w = self.MAX_LEN
        if h > self.MAX_LEN:
            h = self.MAX_LEN
        self.w = w
        self.h = h
        self.xmax = self.xmin + w
        self.ymax = self.ymin + h

    def get_rec(self):
        xmin, ymin = int(self.xmin), int(self.ymin)
        return (xmin, ymin), (xmin+self.w, ymin+self.h)

    def check_occ(self, other):
        """
        检测与其他状态是否碰撞
        :param other: State
        :return: bool
        """

        return max(self.xmin, other.xmin) < min(self.xmax, other.xmax) and \
               max(self.ymin, other.ymin) < min(self.ymax, other.ymax)

    def to_addable_vec(self):
        return np.array([self.xmin, self.ymin])

    def update_left(self, vec):
        self.xmin = vec[0]
        self.ymin = vec[1]
        self.xmax = vec[0]+self.w
        self.ymax = vec[1] + self.h


if __name__ == '__main__':
    A = np.random.randn(4, 3)
    u, s, vh = svd(A, full_matrices=False)
    e = np.random.randn(4, 2)
    u_hat, s_hat, v_hat = rsvd(u, s, vh.T, e)
    tmp = np.dot(u_hat, np.diag(s_hat))
    tmp = np.dot(tmp, v_hat.T)
    A_ = np.hstack((A,e))
    print(np.allclose(A_, tmp))


