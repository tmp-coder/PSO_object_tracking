"""
-*- coding: utf-8 -*-
@Author  : zouzhitao
@Time    : 17-10-5 下午2:21
@File    : pso.py
"""
import numpy as np
from numpy.linalg import svd, norm
from utils import State, bound

# 高斯退火常数
ANNEAL_CONST = 0.2
# 迭代界限
TH = 0.97

class Particle(object):
    def __init__(self, best):
        self.pbest = best
        l, r = best.get_rec()
        self.x = State((0, 0), (r[0]-l[0], r[1]-l[1]))
        self.v = np.random.randn(2)
        self.fit = 0


class Object(object):
    def __init__(self, objmat):
        self.A = objmat
        self.U, self.sigma, self.Vh = svd(objmat, full_matrices=False)
        self.mul_mat = self.U.dot(self.U.T)
        self.tmp = self.mul_mat.dot(objmat)
    def update_space_mat(self,e):
        self.A = np.hstack((self.A, e))
        self.U, self.sigma, self.vh = svd(self.A, full_matrices=False)
        self.mul_mat = self.U.dot(self.U.T)
        self.tmp = self.mul_mat.dot(e)


class Tracker(object):
    # max num of particles
    NUM_PARTICLE = 50

    # PSO const param
    #R1, R2, R3 = np.abs(np.random.randn(3))
    R1, R2, R3 = 1., 1.49, 1.49
    # gauss transition matrix
    GAUSS = np.eye(2, 2)*2
    # anneal cov matrix
    ANNEAL_MAT = np.eye(2)*2
    # max iter num
    MAX_ITER = 50

    def __init__(self, left, right, first_frame):
        self.gbest = State(left, right)
        l, r = self.gbest.get_rec()
        obs = first_frame[l[0]:r[0], l[1]:r[1]]
        self.obj = Object(obs.reshape((obs.size, -1)))
        self.par = [Particle(self.gbest) for _ in range(self.NUM_PARTICLE)]

    def init_particle(self):
        for i in range(self.NUM_PARTICLE):
            x_init = np.random.multivariate_normal(self.par[i].pbest.to_addable_vec(), self.GAUSS)
            self.par[i].x.update_left(x_init)

    def anneal_const(self, num_iter):
        return np.random.multivariate_normal(np.zeros(2), self.ANNEAL_MAT*np.exp(-num_iter*ANNEAL_CONST))

    def fitness_value_unocc(self, s, img):
        # 非重叠实现
        left, right = s.get_rec()
        if not bound(img.shape,left, right):
            return -1e5

        o = img[left[0]:right[0], left[1]:right[1]]
        o = o.reshape((o.size, -1))
        RE = norm(o - self.obj.tmp, 2)
        #print('RE', RE)
        return np.exp(-RE ** 2)
        #return -RE
    def update_particle_unocc(self, idx, iters):
        """
        更新粒子状态,非重叠实现
        :param idx: ith par
        :param iters: ith iterator
        :return:
        """

        pbest = self.par[idx].pbest.to_addable_vec()
        gbest = self.gbest.to_addable_vec()
        x_mat = self.par[idx].x.to_addable_vec()

        self.par[idx].v = self.R1 * (pbest - x_mat) + self.R2 * (gbest - x_mat) + self.anneal_const(iters)

        x_mat = x_mat + self.par[idx].v
        self.par[idx].x.update_left(x_mat)


    def tracking_unocc(self, img):
        """
        非遮挡的tracking
        :param img: ith frame img
        :return:
        """
        self.init_particle()
        gfit = self.fitness_value_unocc(self.gbest, img)
        for par in self.par:
            par.fit = self.fitness_value_unocc(par.x, img)
        for _ in range(self.MAX_ITER):
            for i in range(self.NUM_PARTICLE):
                self.update_particle_unocc(i, _)
                # 可能出现矩阵越界
                try:
                    fit = self.fitness_value_unocc(self.par[i].x, img)
                except :
                    print('---------------errr------------', i)
                    print(fit)
                    print(self.par[i].x)
                    print(img.shape)
                    print(self.par[i].x.get_rec())
                    #Tracker.R1,Tracker.R2,Tracker.R3 = np.abs(np.random.randn(3))
                    #fit = self.fitness_value_unocc(self.par[i].x, img)

                if fit > self.par[i].fit:
                    self.par[i].pbest = self.par[i].x
                    self.par[i].fit = fit
                    if fit > gfit:
                        self.gbest = self.par[i].pbest
                        gfit = fit

            print(_, gfit)


        left, right = self.gbest.get_rec()
        e = img[left[0]:right[0], left[1]:right[1]]
        self.obj.update_space_mat(e.reshape(e.size,1))
        return left, right
















