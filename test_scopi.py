import sys
import time
import numpy as np
import numpy.linalg as npl
import matplotlib
import matplotlib.pyplot as plt

import pyvista as pv ## pip3 install pyvista imageio-ffmpeg

sys.path.append('build/')
import fm
import scopi

### Particles
part = scopi.Particles("particules");
Np = 10
rmin = 1
rmax = 2
mmin = 0.8
mmax = 1.2
xc = 0
yc = 0
zc = 0
rball = 2
part.add_particles_in_ball(Np, rmin, rmax, mmin, mmax, xc, yc, zc, rball)
part.print()

### Obstacles
obs = scopi.Obstacle("objets3D/box.stl",2.5,1)
mesh  = pv.PolyData("objets3D/box.stl")

### Contacts
dxc = 1.2*rmax # rmax + deplacement maximal pendant un dt
cont = scopi.Contacts(dxc)
cont.compute_contacts(part)
cont.print()

### Projection
maxiter = 40000
dmin = 0.1
dt = 0.1
rho = 0.2
tol = 1.0e-2
proj = scopi.Projection(maxiter, rho, dmin, tol, dt)
proj.run(part,cont)

### Move particles
part.move(dt)
part.print()

### Plot the spheres
pl = pv.Plotter()
sphere = pv.Sphere(theta_resolution=12,phi_resolution=12)
pmesh = pv.PolyData(part.get_positions())
pmesh["r"] = part.get_r()
spheres = pmesh.glyph(scale="r", factor=2, geom=sphere)
spheres_actor = pl.add_mesh(spheres,name="spheres")
obs_actor = pl.add_mesh(mesh,name="obs",opacity=0.5)
cpos = pl.show(auto_close=False)
