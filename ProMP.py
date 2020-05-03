#!/usr/bin/env python
# coding: utf-8

import numpy as np

B = 10
c = np.linspace(0.0, 1.0, B)
h = -(1.0/(B-1))**2/(2*np.log(0.3))
hz = 100

def radial_basis(phase,centers,bandwidth):
    bases = np.exp(-(np.repeat(phase[...,np.newaxis],centers.shape,-1)-centers)**2/(2*bandwidth)).T
    bases /= bases.sum(axis=0)
    return bases.T

#data needs to be dimensions x time stamps
def learn_weights(data, phi, lamda=1e-6):
    w = np.linalg.solve(np.dot(phi.T,phi)+lamda*np.eye(B),np.dot(phi.T,data[0:,:].T)).T
    return w

def get_phase(t):
    phase = t-np.min(t)
    phase /= np.max(phase)
    return phase

def learn_weight_distribution(trajectories):
    ws = np.array([learn_weights(d,radial_basis(get_phase(d[0,:]),c,h)).flatten() for d in trajectories])
    mu = np.mean(ws,axis=0)
    sigma = np.cov(ws.T)
    return mu, sigma

def get_traj_distribution(mu_w, sigma_w, des_duration=1.0):
    des_t = np.arange(0.0,des_duration,1.0/hz)
    z = get_phase(des_t)
    phi = radial_basis(z,c,h)
    D = 2
    psi = np.kron(np.eye(int(mu_w.shape[0]/B),dtype=int),phi)
    mu = np.dot(psi,mu_w)
    sigma = np.dot(np.dot(psi,sigma_w),psi.T)
    return mu, sigma

def conditioning(sigma_w, sigma_y, t, mu_w, y_t):
    des_t = np.arange(0.0,1.0,1.0/hz)
    z = get_phase(des_t)
    phi = radial_basis(z, c, h)
    phi_t = phi[t].reshape(B, 1)
    psi_t = np.kron(np.eye(int(mu_w.shape[0]/B),dtype=int),phi_t)

    L = sigma_w.dot(psi_t).dot(np.linalg.inv(sigma_y + psi_t.T.dot(sigma_w.dot(psi_t))))
    
    mu_w_new = mu_w + L.dot(y_t - psi_t.T.dot(mu_w))
    
    sigma_w_new = sigma_w - L.dot(psi_t.T.dot(sigma_w))
    
    return mu_w_new, sigma_w_new

# takes in flattened version (all dimensions in 1D array), returns stacked based on 7 dimensions
def stack(mean_traj):
    stacked_mean_traj = [[],[],[],[],[],[],[]]
    for i in range(7):
        stacked_mean_traj[i] = mean_traj[i * mean_traj.shape[0] // 7: (i+1) * mean_traj.shape[0] // 7]
    return np.reshape(stacked_mean_traj, (7,mean_traj.shape[0] // 7))

# takes in group of trajectories and returns a promp
def make_ProMP (grouped_traj):
    mean_w, cov_w  = learn_weight_distribution(grouped_traj)
    mean_traj, cov_traj = get_traj_distribution(mean_w, cov_w)
    return (mean_traj, cov_traj)

def sample (mean_traj, cov_traj):
    s = np.random.multivariate_normal(mean_traj, cov_traj, 1).T
    return stack(s)



