import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import interpn
import pyvista
from time import time
sys.path.append('build/')
import fm

#mesh = pyvista.PolyData('empty_hourglass_subdivide3.obj')
mesh  = pyvista.PolyData('empty_hourglass_subdivide3.stl')
print(mesh)
print("bounds : ",mesh.bounds)
print("points : ",mesh.points)
print("point_arrays : ",mesh.point_arrays)
print("cell_arrays : ",mesh.cell_arrays)

ind = np.where((mesh.cell_arrays['MaterialIds']==0))[0]
print("MaterialIds==0 ind.shape : ",ind.shape)
print("mesh.number_of_faces : ",mesh.number_of_faces)
print("mesh.faces.shape : ",mesh.faces.shape)

#mesh.plot(show_edges=True, color=True)
h = 0.1
xmin,xmax,ymin,ymax,zmin,zmax = mesh.bounds
xmin = np.floor(xmin)
ymin = np.floor(ymin)
zmin = np.floor(zmin)
xmax = np.ceil(xmax)
ymax = np.ceil(ymax)
zmax = np.ceil(zmax)
print("domain : [",xmin,",",xmax,"]x[",ymin,",",ymax,"]x[",zmin,",",zmax,"]")

x = np.arange(xmin-2*h,xmax+2*h,step=h)
y = np.arange(ymin-2*h,ymax+2*h,step=h)
z = np.arange(zmin-2*h,zmax+2*h,step=h)
indx = (np.round((mesh.points[:,0]-x[0])/h)).astype(np.int)
indy = (np.round((mesh.points[:,1]-y[0])/h)).astype(np.int)
indz = (np.round((mesh.points[:,2]-z[0])/h)).astype(np.int)
nx = x.shape[0]
ny = y.shape[0]
nz = z.shape[0]
print("nx = ",nx," ny = ",ny," nz = ",nz)

img = 1.0e20*np.ones((nx,ny,nz))
img[indx,indy,indz] = 0

narrow_band = np.vstack(np.where((img==0))).T
t0 = time()
fm.compute_distance(h, np.amax(img), img, narrow_band)
print("time to compute distance : ",time()-t0)

dims = (nx,ny,nz)
spacing = (h, h, h)
grid = pyvista.UniformGrid(dims,spacing)
grid.point_arrays["wall"] = img.flatten(order="F")
slices = grid.slice_orthogonal()
cmap = plt.cm.get_cmap("viridis")#, 4)
slices.plot(cmap=cmap)
