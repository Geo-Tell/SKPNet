#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import numpy as np
import open3d as o3d

class Spherical(object):
  '''球坐标系'''
  def __init__(self, radial = 1.0, polar = 0.0, azimuthal = 0.0):
    self.radial = radial
    self.polar = polar
    self.azimuthal = azimuthal
  def toCartesian(self):
    '''转直角坐标系'''
    r = math.sin(self.polar) * self.radial
    x = math.cos(self.azimuthal) * r
    y = math.sin(self.azimuthal) * r
    z = math.cos(self.polar) * self.radial
    return x, y, z

def splot(N_polar, N_azimuthal):
  s = Spherical()
  points = []
  deta_polar = 2 * math.pi / N_polar
  deta_azimuthal = 2 * math.pi / N_azimuthal
  s.polar = deta_polar
  s.azimuthal = 0
  for polar_i in range(N_polar-1):
    for azi_i in range(N_azimuthal):
      points.append(s.toCartesian())
      s.azimuthal += deta_azimuthal
    if polar_i == N_polar/2 - 1:
      s.polar += 2* deta_polar
    else:
      s.polar += deta_polar

  points.append([0,0,1])
  points.append([0,0,-1])
  return np.array(points).T

v = splot(16,16)
print(v.shape)

np.save('sphere%d' % v.shape[1], v.transpose())
p = o3d.utility.Vector3dVector(v)
pc = o3d.geometry.PointCloud(p)
o3d.visualization.draw_geometries([pc])


# for point in splot(100):
#   print("%f %f %f" % point)