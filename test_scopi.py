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

n = 100

rmin = 0.1
rmax = 0.2

xc = -4
yc = 0
zc = 7
rball = 6
p = scopi.Particles("sand in ball");
p.add_particles_in_ball(n, rmin, rmax, xc, yc, zc, rball)
# xmin = 0
# xmax = 10
# ymin = 1
# ymax = 11
# zmin = 2
# zmax = 12
#p = scopi.Particles("sand in box");
#p.add_particles_in_box(n,rmin,rmax,xmin,xmax,ymin,ymax,zmin,zmax)

#p.print()

xyzr = p.xyzr()
print("(init) xyzr.data = ",xyzr.data)

vap = np.zeros((xyzr.shape[0],3))
#vap[:,2] = -1
vap[:,0] = xyzr[:,0]-xc
vap[:,1] = xyzr[:,1]-yc
vap[:,2] = xyzr[:,2]-zc
vap_norm = npl.norm(vap, axis=1)
vap[:,0] = -vap[:,0]/vap_norm
vap[:,1] = -vap[:,1]/vap_norm
vap[:,2] = -vap[:,2]/vap_norm

#save_movie = False

p = pv.Plotter()
sphere = pv.Sphere(theta_resolution=12,phi_resolution=12)
mesh = pv.PolyData(xyzr[:,:3])
mesh["r"] = xyzr[:,3]
spheres = mesh.glyph(scale="r", factor=2, geom=sphere)
spheres_actor = p.add_mesh(spheres,name="spheres")
# arrow = pv.Arrow()
# mesh['vap'] = vap
# arrows = mesh.glyph(orient="vap", scale=False, factor=0.5, geom=arrow)
# p.add_mesh(arrows)
# # hourglass  = pv.PolyData('empty_hourglass_subdivide3.obj') ## BUG VTK pour lire le fichier (1ere colonne caractere 'v')
# # hourglass  = pv.PolyData('empty_hourglass_subdivide3.stl')
# hourglass  = pv.PolyData('empty_hourglass_subdivide1.stl')
# print("hourglass bounds : ",hourglass.bounds)
# print("hourglass.points.shape = ",hourglass.points.shape," hourglass.faces.shape = ",hourglass.faces.shape)
# p.add_mesh(hourglass, name="hourglass", opacity=0.5, color=True)
cpos = p.show(auto_close=False)


maxiter = 100000000
dmin = 0.1
dt = 0.1
rho = 0.2
tol = 1.0e-2
t = 0
tf = 5

c = scopi.Contacts()
dxc = 1.2*rmax # rmax + deplacement maximal pendant un dt

invMass = 1/np.ones( (vap.shape[1]*vap.shape[0],) )

while (t<tf-0.5*dt):

    print("\n\n t = ",t,"\n")

    contacts = c.compute_contacts(xyzr, dxc)
    try:
        nc = contacts.shape[0]
    except:
        nc = -1

    if (nc>0):
        print("  nb of contacts = ",nc)
        proj = scopi.Projection()
        res = proj.run(
            xyzr,
            contacts,
            vap.reshape( (vap.shape[1]*vap.shape[0],) ),
            contacts[:,2],
            invMass,
            maxiter,
            rho,
            dmin,
            tol,
            dt)
        new_V = res.reshape((xyzr.shape[0],3))
    else:
        new_V = vap

    xyzr[:,0] =  xyzr[:,0]+dt*new_V[:,0]
    xyzr[:,1] =  xyzr[:,1]+dt*new_V[:,1]
    xyzr[:,2] =  xyzr[:,2]+dt*new_V[:,2]
    new_sphere = pv.Sphere(theta_resolution=12,phi_resolution=12)
    new_mesh = pv.PolyData(xyzr[:,:3])
    new_mesh["r"] = xyzr[:,3]
    new_spheres = new_mesh.glyph(scale="r", factor=2, geom=new_sphere)
    p.add_mesh(new_spheres,name="spheres")
    # p.mesh.points = new_spheres.points
    # p.mesh.faces = new_spheres.faces  ##
    p.render()

    vap[:,0] = xyzr[:,0]-xc
    vap[:,1] = xyzr[:,1]-yc
    vap[:,2] = xyzr[:,2]-zc
    vap_norm = npl.norm(vap, axis=1)
    vap[:,0] = -vap[:,0]/vap_norm
    vap[:,1] = -vap[:,1]/vap_norm
    vap[:,2] = -vap[:,2]/vap_norm

    t += dt

    if (t>=tf-0.5*dt):
        p_end = pv.Plotter()
        p_end.camera_position = cpos
        new_sphere = pv.Sphere(theta_resolution=12,phi_resolution=12)
        new_mesh = pv.PolyData(xyzr[:,:3])
        new_mesh["r"] = xyzr[:,3]
        new_spheres = new_mesh.glyph(scale="r", factor=2, geom=new_sphere)
        p_end.add_mesh(new_spheres)
        #p_end.add_mesh(hourglass, opacity=0.5, color=True)
        p_end.show(auto_close=False)
