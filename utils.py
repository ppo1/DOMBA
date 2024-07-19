import torch
from torch.distributions import normal
import numpy as np

def no_avg(a,b):
    return a

def harmonic_avg(a,b):
    return torch.log(2. / ((1. / torch.exp(a)) + (1. / torch.exp(b))))

def general_avg(a,b, n=-2):
    return torch.log(((torch.exp(a*n) + torch.exp(b*n)) / 2.)) * 1./n

def get_avg(n):
  return lambda a,b: general_avg(a,b,n)

def arithmetic_avg(a,b):
    return torch.log((torch.exp(a) + torch.exp(b))/2)

def min_avg(a,b):
    return torch.min(a,b)

def geometric_avg(a,b):
    return (a + b) / 2

def l2p(l):
    norm_l = torch.log(torch.sum(torch.exp(l), dim=-1, keepdim=True))
    return torch.exp(l - norm_l)

def reny_d(p0,p1):
    return 0.5*torch.log(torch.maximum(torch.sum(p0**2 / p1, dim=-1, keepdim=True), torch.sum(p1**2 / p0, dim=-1,keepdim=True)))

def submix(l0,l1,lb, beta):
    p0 = l2p(l0)
    p1 = l2p(l1)
    pb = l2p(lb)
    res = torch.zeros(lb.shape).to(lb.device)
    for g in [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 0.9, 1]:
      p0b = g * p0 + (1-g) * pb
      p1b = g * p1 + (1-g) * pb
      rd = reny_d(p0b, p1b)
      p_all = 0.5*p0b + 0.5*p1b
      res = torch.where(rd <= beta, torch.log(p_all), res)
    return res

def submix_1e2(l0,l1,lb):
  return submix(l0,l1,lb, 0.01)

def submix_3e1(l0,l1,lb):
  return submix(l0,l1,lb, 0.3)