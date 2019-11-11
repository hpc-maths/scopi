import sys
import time
import numpy as np
import numpy.linalg as npl
import matplotlib
import matplotlib.pyplot as plt
#from matplotlib.collections import EllipseCollection
#from mpl_toolkits.mplot3d import Axes3D

import pyvista as pv ## pip3 install pyvista imageio-ffmpeg

sys.path.append('build/')
import scopi

Np = 100

rmin = 0.1
rmax = 0.2

mmin = 0.8
mmax = 1.2

xc = -4
yc = 0
zc = 7
rball = 6
p = scopi.Particles("sand in ball");
p.add_particles_in_ball(Np, rmin, rmax, mmin, mmax, xc, yc, zc, rball)
# xmin = xc-2
# xmax = xc+2
# ymin = yc-2
# ymax = yc+2
# zmin = zc-2
# zmax = zc+2
# p.add_particles_in_box(Np,rmin,rmax,mmin,mmax,xmin,xmax,ymin,ymax,zmin,zmax)
# p.print()

#save_movie = False
pl = pv.Plotter()
sphere = pv.Sphere(theta_resolution=12,phi_resolution=12)
mesh = pv.PolyData(p.get_positions())
mesh["r"] = p.get_r()
spheres = mesh.glyph(scale="r", factor=2, geom=sphere)
spheres_actor = pl.add_mesh(spheres,name="spheres")
# # arrow = pv.Arrow()
# # mesh['vap'] = vap
# # arrows = mesh.glyph(orient="vap", scale=False, factor=0.5, geom=arrow)
# # p.add_mesh(arrows)
# # # hourglass  = pv.PolyData('empty_hourglass_subdivide3.obj') ## BUG VTK pour lire le fichier (1ere colonne caractere 'v')
# # # hourglass  = pv.PolyData('empty_hourglass_subdivide3.stl')
# # hourglass  = pv.PolyData('empty_hourglass_subdivide1.stl')
# # print("hourglass bounds : ",hourglass.bounds)
# # print("hourglass.points.shape = ",hourglass.points.shape," hourglass.faces.shape = ",hourglass.faces.shape)
# # p.add_mesh(hourglass, name="hourglass", opacity=0.5, color=True)
cpos = pl.show(auto_close=False)

maxiter = 40000
dmin = 0.1
dt = 0.1
rho = 0.2
tol = 1.0e-2
t = 0
tf = 5
dxc = 1.2*rmax # rmax + deplacement maximal pendant un dt
c = scopi.Contacts(dxc)
proj = scopi.Projection(maxiter, rho, dmin, tol, dt)

withplot = False

while (t<tf-0.5*dt):

    print("\n\n t = ",t,"\n")

    vap = np.zeros((Np,3))
    pos = p.get_positions()
    vap[:,0] = pos[:,0]-xc
    vap[:,1] = pos[:,1]-yc
    vap[:,2] = pos[:,2]-zc
    vap_norm = npl.norm(vap, axis=1)
    vap[:,0] = -vap[:,0]/vap_norm
    vap[:,1] = -vap[:,1]/vap_norm
    vap[:,2] = -vap[:,2]/vap_norm
    p.set_vap(vap)

    c.compute_contacts(p)

    proj.run(p,c)

    p.move(dt)

    if (withplot==True):
        new_sphere = pv.Sphere(theta_resolution=12,phi_resolution=12)
        new_mesh = pv.PolyData(p.get_positions())
        new_mesh["r"] = p.get_r()
        new_spheres = new_mesh.glyph(scale="r", factor=2, geom=new_sphere)
        pl.add_mesh(new_spheres,name="spheres")
        # p.mesh.points = new_spheres.points
        # p.mesh.faces = new_spheres.faces  ##
        pl.render()

    t += dt

    if (t>=tf-0.5*dt):
        p_end = pv.Plotter()
        p_end.camera_position = cpos
        new_sphere = pv.Sphere(theta_resolution=12,phi_resolution=12)
        new_mesh = pv.PolyData(p.get_positions())
        new_mesh["r"] = p.get_r()
        new_spheres = new_mesh.glyph(scale="r", factor=2, geom=new_sphere)
        p_end.add_mesh(new_spheres)
        #p_end.add_mesh(hourglass, opacity=0.5, color=True)
        p_end.show(auto_close=False)
