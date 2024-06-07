import pyvista as pv
from pyvista import examples
import numpy as np
import json
import glob
import sys
import time

prefix = "../../../build/Results/"

files = np.sort(glob.glob(prefix+"/*.json"))
print(files)

plotter = pv.Plotter()
plotter.show_grid()
plotter.show_bounds()

# light_types = [light.light_type for light in plotter.renderer.lights]
# Remove from plotters so output is not produced in docs
# pv.plotting._ALL_PLOTTERS.clear()
# print("light_types =",light_types)


actors = []
geometries = []


it = 0
for file in files[::1]:

    # plotter = pv.Plotter(off_screen=True)
    # plotter.clear()

    with open(file) as json_file:
        print("read json file :",file)
        data = json.load(json_file)

    objects = data["objects"]
    contacts = data["contacts"]

    positions = []

    if (it==0):
        for obj in objects:
            positions.append(obj["position"])
            v = np.zeros( (len(obj["position"]),) )
            v[0] = 1
            orientation = np.array(obj["rotation"]).reshape((len(obj["position"]),len(obj["position"])))@v
            if (obj["type"] == "sphere"):
                if (len(obj["position"])==2):  # 2D
                    geom = pv.ParametricSuperEllipsoid(
                        xradius=obj["radius"],
                        yradius=obj["radius"],
                        zradius=0,
                        n1=1,
                        n2=1,
                        center=(obj["position"][0],obj["position"][1],0),
                        direction=(orientation[0],orientation[1],0)
                    )
                    geom["e"] = np.ones((np.array(geom.points).shape[0],))
                    geom["n"] = np.ones((np.array(geom.points).shape[0],))
                    geom["e+n"] = 2*np.ones((np.array(geom.points).shape[0],))
                    # geom = pv.Sphere(
                    #     radius=obj["radius"],
                    #     center=(obj["position"][0],obj["position"][1],0),
                    #     direction=(orientation[0],orientation[1],0)
                    #     )

                else: # 3D
                    geom = pv.Sphere(
                        radius=obj["radius"],
                        center=(obj["position"][0],obj["position"][1],obj["position"][2]),
                        direction=(orientation[0],orientation[1],orientation[2])
                        )
                    geom["e"] = np.ones((np.array(geom.points).shape[0],))
                    geom["n"] = np.ones((np.array(geom.points).shape[0],))
                    geom["e+n"] = 2*np.ones((np.array(geom.points).shape[0],))


            elif (obj["type"] == "superellipsoid"):
                if (len(obj["position"])==2):  # 2D
                    geom = pv.ParametricSuperEllipsoid(
                        xradius=obj["radius"][0],
                        yradius=obj["radius"][1],
                        zradius=0,
                        n1=1,
                        n2=obj["squareness"][0],
                        center=(obj["position"][0],obj["position"][1],0),
                        direction=(orientation[0],orientation[1],0)
                        )
                    geom["e"] = obj["squareness"][0]*np.ones((np.array(geom.points).shape[0],))
                    geom["n"] = np.ones((np.array(geom.points).shape[0],))
                    geom["e+n"] = obj["squareness"][0]*np.ones((np.array(geom.points).shape[0],))+np.ones((np.array(geom.points).shape[0],))


                else: # 3D
                    geom = pv.ParametricSuperEllipsoid(
                        xradius=obj["radius"][0],
                        yradius=obj["radius"][1],
                        zradius=obj["radius"][2],
                        n1=obj["squareness"][0],
                        n2=obj["squareness"][1],
                        center=(obj["position"][0],obj["position"][1],obj["position"][2]),
                        direction=(orientation[0],orientation[1],orientation[2])
                        )
                    geom["e"] = obj["squareness"][0]*np.ones((np.array(geom.points).shape[0],))
                    geom["n"] = obj["squareness"][1]*np.ones((np.array(geom.points).shape[0],))
                    geom["e+n"] = obj["squareness"][0]*np.ones((np.array(geom.points).shape[0],))+obj["squareness"][1]*np.ones((np.array(geom.points).shape[0],))
            elif (obj["type"] == "plane"):
                if (len(obj["position"])==2):  # 2D
                    geom = pv.Plane(
                        center=(obj["position"][0],obj["position"][1],0),
                        direction=(orientation[0],orientation[1],0)
                        )
                    geom["e"] = -1*np.ones((np.array(geom.points).shape[0],))
                    geom["n"] = -1*np.ones((np.array(geom.points).shape[0],))
                    geom["e+n"] = -2*np.ones((np.array(geom.points).shape[0],))

            geometries.append(geom)
            plotter.add_mesh(geom, show_scalar_bar=False, scalars="e+n",clim=[1.2, 2.0])#, specular=1, specular_power=15, smooth_shading=True, scalars="e+n",clim=[1.2, 2.0])

    else: # it>1

        for io,obj in enumerate(objects):
            # print(io)
            positions.append(obj["position"])
            v = np.zeros( (len(obj["position"]),) )
            v[0] = 1
            orientation = np.array(obj["rotation"]).reshape((len(obj["position"]),len(obj["position"])))@v
            if (obj["type"] == "sphere"):
                if (len(obj["position"])==2):  # 2D
                    geom = pv.ParametricSuperEllipsoid(
                        xradius=obj["radius"],
                        yradius=obj["radius"],
                        zradius=0,
                        n1=1,
                        n2=1,
                        center=(obj["position"][0],obj["position"][1],0),
                        direction=(orientation[0],orientation[1],0)
                    )
                    geom["e"] = np.ones((np.array(geom.points).shape[0],))
                    geom["n"] = np.ones((np.array(geom.points).shape[0],))
                    geom["e+n"] = 2*np.ones((np.array(geom.points).shape[0],))
                    # geom = pv.Sphere(
                    #     radius=obj["radius"],
                    #     center=(obj["position"][0],obj["position"][1],0),
                    #     direction=(orientation[0],orientation[1],0)
                    #     )

                else: # 3D
                    geom = pv.Sphere(
                        radius=obj["radius"],
                        center=(obj["position"][0],obj["position"][1],obj["position"][2]),
                        direction=(orientation[0],orientation[1],orientation[2])
                        )
                    geom["e"] = np.ones((np.array(geom.points).shape[0],))
                    geom["n"] = np.ones((np.array(geom.points).shape[0],))
                    geom["e+n"] = 2*np.ones((np.array(geom.points).shape[0],))

            elif (obj["type"] == "superellipsoid"):
                if (len(obj["position"])==2):  # 2D
                    geom = pv.ParametricSuperEllipsoid(
                        xradius=obj["radius"][0],
                        yradius=obj["radius"][1],
                        zradius=0,
                        n1=1,
                        n2=obj["squareness"][0],
                        center=(obj["position"][0],obj["position"][1],0),
                        direction=(orientation[0],orientation[1],0)
                        )
                    geom["e"] = obj["squareness"][0]*np.ones((np.array(geom.points).shape[0],))
                    geom["n"] = np.ones((np.array(geom.points).shape[0],))
                    geom["e+n"] = obj["squareness"][0]*np.ones((np.array(geom.points).shape[0],))+np.ones((np.array(geom.points).shape[0],))


                else: # 3D
                    geom = pv.ParametricSuperEllipsoid(
                        xradius=obj["radius"][0],
                        yradius=obj["radius"][1],
                        zradius=obj["radius"][2],
                        n1=obj["squareness"][0],
                        n2=obj["squareness"][1],
                        center=(obj["position"][0],obj["position"][1],obj["position"][2]),
                        direction=(orientation[0],orientation[1],orientation[2])
                        )
                    geom["e"] = obj["squareness"][0]*np.ones((np.array(geom.points).shape[0],))
                    geom["n"] = obj["squareness"][1]*np.ones((np.array(geom.points).shape[0],))
                    geom["e+n"] = obj["squareness"][0]*np.ones((np.array(geom.points).shape[0],))+obj["squareness"][1]*np.ones((np.array(geom.points).shape[0],))

            elif (obj["type"] == "plane"):
                if (len(obj["position"])==2):  # 2D
                    geom = pv.Plane(
                        center=(obj["position"][0],obj["position"][1],0),
                        direction=(orientation[0],orientation[1],0)
                        )
                    geom["e"] = -np.ones((np.array(geom.points).shape[0],))
                    geom["n"] = -np.ones((np.array(geom.points).shape[0],))
                    geom["e+n"] = -2*np.ones((np.array(geom.points).shape[0],))

            #print("test1 = ",np.linalg.norm(np.array(geom.points)-np.array(geometries[io].points)))
            geometries[io].points = geom.points
            #print("test2 = ",np.linalg.norm(np.array(geom.points)-np.array(geometries[io].points)))

            # print(geom["e+n"] )

    # print("positions = ",positions)
    # print("geometries = ",geometries)

    # if contacts is not None:
    #     for ic, contact in enumerate(contacts):
    #         if (len(contact["pi"])==2): # dim=2
    #             contact["pi"].append(0)
    #             contact["pj"].append(0)
    #             contact["nij"].append(0)
    #         pvpti = pv.PolyData(np.array([contact["pi"]]))
    #         pvptj = pv.PolyData(np.array([contact["pj"]]))
    #         pvpti["normal"] = -np.asarray([contact["nij"]])
    #         pvptj["normal"] = np.array([contact["nij"]])
    #         plotter.add_mesh(pvpti.glyph(orient="normal",factor=0.05, geom=pv.Arrow()),color="blue",name=f"ni_{ic}")
    #         plotter.add_mesh(pvptj.glyph(orient="normal",factor=0.05, geom=pv.Arrow()),color="blue",name=f"nj_{ic}")
    #

        # plotter.camera_position = 'xy'
        # plotter.camera.SetParallelProjection(True)

    if (it == 0):
        # plotter.show(auto_close=False, cpos="xy")
        plotter.show(auto_close=False, cpos="xy",screenshot=file.replace(".json",".png"))#,title=str(it))
        # plotter.render()
        # plotter.show(auto_close=False, screenshot=file.replace(".json",".png"))
        # plotter.write_frame()

        # Open a movie file
        plotter.open_movie(prefix+"/film.mp4")

    else:
        # plotter.show_grid()
        # plotter.show_bounds()
        plotter.write_frame()
        plotter.render()
        # plotter.show(auto_close=True, cpos="xy",screenshot=file.replace(".json",".png"),title=str(it))
        time.sleep(0.001)

    it+=1

plotter.close()
sys.exit()





with open('build/scopi_objects_0599.json') as json_file:
    data = json.load(json_file)

objects = data["objects"]
contacts = data["contacts"]

plotter = pv.Plotter()

positions = []
geometries = []

for obj in objects:

    positions.append(obj["position"])
    v = np.zeros( (len(obj["position"]),) )
    v[0] = 1
    orientation = np.array(obj["rotation"]).reshape((len(obj["position"]),len(obj["position"])))@v
    if (obj["type"] == "sphere"):
        if (len(obj["position"])==2):  # 2D
            geom = pv.ParametricSuperEllipsoid(
                xradius=obj["radius"],
                yradius=obj["radius"],
                zradius=0,
                n1=1,
                n2=1,
                center=(obj["position"][0],obj["position"][1],0),
                direction=(orientation[0],orientation[1],0)
            )
            # geom = pv.Sphere(
            #     radius=obj["radius"],
            #     center=(obj["position"][0],obj["position"][1],0),
            #     direction=(orientation[0],orientation[1],0)
            #     )

        else: # 3D
            geom = pv.Sphere(
                radius=obj["radius"],
                center=(obj["position"][0],obj["position"][1],obj["position"][2]),
                direction=(orientation[0],orientation[1],orientation[2])
                )
    elif (obj["type"] == "superellipsoid"):
        if (len(obj["position"])==2):  # 2D
            geom = pv.ParametricSuperEllipsoid(
                xradius=obj["radius"][0],
                yradius=obj["radius"][1],
                zradius=0,
                n1=1,
                n2=obj["squareness"][0],
                center=(obj["position"][0],obj["position"][1],0),
                direction=(orientation[0],orientation[1],0)
                )
        else: # 3D
            geom = pv.ParametricSuperEllipsoid(
                xradius=obj["radius"][0],
                yradius=obj["radius"][1],
                zradius=obj["radius"][2],
                n1=obj["squareness"][0],
                n2=obj["squareness"][1],
                center=(obj["position"][0],obj["position"][1],obj["position"][2]),
                direction=(orientation[0],orientation[1],orientation[2])
                )

    geometries.append(geom)
    plotter.add_mesh(geom, specular=1, specular_power=15,smooth_shading=True, show_scalar_bar=False)

print("positions = ",positions)
print("geometries = ",geometries)

for contact in contacts:
    contact["pi"].append(0)
    contact["pj"].append(0)
    contact["nij"].append(0)
    pvpti = pv.PolyData(np.array([contact["pi"]]))
    pvptj = pv.PolyData(np.array([contact["pj"]]))
    pvpti["normal"] = -np.asarray([contact["nij"]])
    pvptj["normal"] = np.array([contact["nij"]])
    plotter.add_mesh(pvpti.glyph(orient="normal",factor=0.01, geom=pv.Arrow()),color="pink",name="ni")
    plotter.add_mesh(pvptj.glyph(orient="normal",factor=0.01, geom=pv.Arrow()),color="blue",name="nj")

plotter.camera_position = 'xy'
plotter.camera.SetParallelProjection(True)

plotter.show_grid()
plotter.show_bounds()

plotter.show()


"""
## 3D ##

## rx ry rz n e
param_superellipsoids = np.array([[0.25, 0.25, 0.25, 1,   1],
                                  [1,    0.25, 0.25, 1,   1],
                                  [0.25, 1,    0.25, 1,   1],
                                  [0.25, 0.25, 1,    1,   1],
                                  [0.25, 0.5,  0.75, 0.9, 0.8]])

geoms = [pv.ParametricSuperEllipsoid(xradius=rx, yradius=ry, zradius=rz, n1=n1, n2=n2) for rx, ry, rz, n1, n2 in param_superellipsoids]

## r
param_spheres = np.array([[0.1],[0.2],[0.3]])
geoms += [pv.Sphere(radius=r) for r in param_spheres]


pos = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1],  [2,2,2],[3,3,3],[4,4,4]])
mesh = pv.StructuredGrid(pos[:,0], pos[:,1], pos[:,2])
print("mesh.points = ",mesh.points)

# wx wy wz theta(radians)
rotations = np.array([[0,0,1, 0],[0,0,1, np.pi/4],[0,0,1, 0],[0,0,1, 0],[0,0,1, 0],    [0,0,1, 0],[0,0,1, 0],[0,0,1, 0]])
orientations = []
for wx,wy,wz,th in rotations:
    q0,q1,q2,q3 = np.array([
        np.cos(th/2),
        wx*np.sin(th/2),
        wy*np.sin(th/2),
        wz*np.sin(th/2)])
    Mr = np.array([[ q0**2+q1**2-q2**2-q3**2, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2],
                   [ 2*q1*q2+2*q0*q3, q0**2-q1**2+q2**2-q3**2, 2*q2*q3-2*q0*q1],
                   [ 2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, q0**2-q1**2-q2**2+q3**2]])
    orientations.append(Mr@np.array([1,0,0]))
orientations = np.array(orientations)
print("orientations = ",orientations)

# construct the glyphs on top of the mesh; don't scale by scalars now
mesh.point_arrays['scalars'] = np.arange(pos.shape[0])
mesh.point_arrays['vactors'] = orientations
glyphs = mesh.glyph(orient=True, geom=geoms, indices=np.arange(pos.shape[0]), scale=False, factor=1, rng=(0, pos.shape[0]-1))
# orient (bool) – Use the active vectors array to orient the glyphs
# scale (bool) – Use the active scalars to scale the glyphs
# factor (float) – Scale factor applied to scaling array
# geom (vtk.vtkDataSet or tuple(vtk.vtkDataSet), optional) – The geometry to use for the glyph. If missing, an arrow glyph is used. If a sequence, the datasets inside define a table of geometries to choose from based on scalars or vectors. In this case a sequence of numbers of the same length must be passed as indices. The values of the range (see rng) affect lookup in the table.
# indices (tuple(float), optional) – Specifies the index of each glyph in the table for lookup in case geom is a sequence. If given, must be the same length as geom. If missing, a default value of range(len(geom)) is used. Indices are interpreted in terms of the scalar range (see rng). Ignored if geom has length 1.
# tolerance (float, optional) – Specify tolerance in terms of fraction of bounding box length. Float value is between 0 and 1. Default is None. If absolute is True then the tolerance can be an absolute distance. If None, points merging as a preprocessing step is disabled.
# absolute (bool, optional) – Control if tolerance is an absolute distance or a fraction.
# clamping (bool) – Turn on/off clamping of “scalar” values to range.
# rng (tuple(float), optional) – Set the range of values to be considered by the filter when scalars values are provided.
# progress_bar (bool, optional) – Display a progress bar to indicate progress.


# create plotter and add our glyphs with some nontrivial lighting
plotter = pv.Plotter()
plotter.add_mesh(glyphs, specular=1, specular_power=15,
                 smooth_shading=True, show_scalar_bar=False)
plotter.show_grid()
plotter.show_bounds()
plotter.show()

# geom = pv.Sphere(radius=3.14)
#
# ## positions x,y,z
# xyz = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]])
#
# ## parametres des superellipsoides rx, ry, rz, n et e
# params = np.array([[0.1, 0.1, 0.1, 0.56821334, 1.99649769],
#                    [0.1, 0.6, 0.1, 0.08247844, 1.83758874],
#                    [0.1, 0.1, 0.6, 0.49598881, 0.83495047],
#                    [0.05, 0.1, 0.1, 0.52442129, 0.89600688],
#                    [0.2, 0.2, 0.1, 1.92212387, 0.78096621]])
# geoms = [ pv.ParametricSuperEllipsoid(xradius=rx, yradius=ry, zradius=rz, n1=n, n2=e) for rx, ry, rz, n, e in params ]
#
# mesh = pv.PolyData(xyz)
#
#
# # construct the glyphs on top of the mesh; don't scale by scalars now
# glyphs = mesh.glyph(geom=geoms,indices=np.arange(params.shape[0]))#, scale=False)
#
# plotter = pv.Plotter()
# plotter.add_mesh(glyphs, specular=1, specular_power=15,
#                  smooth_shading=True, show_scalar_bar=False)
# plotter.show()

#
#
# # rng_int = rng.integers(0, N, size=x.size)
# rng_int = np.array([4, 1, 2, 0, 4, 0, 1, 4, 3, 1, 1, 3, 3, 4, 3, 4, 4, 3, 3, 2, 2, 1, 1, 1, 2, 0, 3])
j#
#
# # get dataset for the glyphs: supertoroids in xy plane
# # use N random kinds of toroids over a mesh with 27 points
# N = 5
# values = np.arange(N)  # values for scalars to look up glyphs by
#
#
# # taken from:
# # rng = np.random.default_rng()
# # params = rng.uniform(0.5, 2, size=(N, 2))  # (n1, n2) parameters for the toroids
# params = np.array([[1.56821334, 0.99649769],
#                    [1.08247844, 1.83758874],
#                    [1.49598881, 0.83495047],
#                    [1.52442129, 0.89600688],
#                    [1.92212387, 0.78096621]])
#
#
#
#
#
# # get dataset where to put glyphs
# x,y,z = np.mgrid[:3, :3, :3]
# mesh = pv.StructuredGrid(x, y, z)
#
# # add random scalars
# # rng_int = rng.integers(0, N, size=x.size)
# rng_int = np.array([4, 1, 2, 0, 4, 0, 1, 4, 3, 1, 1, 3, 3, 4, 3, 4, 4,
#                     3, 3, 2, 2, 1, 1, 1, 2, 0, 3])
# mesh.point_arrays['scalars'] = rng_int
#
# # construct the glyphs on top of the mesh; don't scale by scalars now
# glyphs = mesh.glyph(geom=geoms, indices=values, scale=False, factor=0.3, rng=(0, N-1))
#
# # create plotter and add our glyphs with some nontrivial lighting
# plotter = pv.Plotter()
# plotter.add_mesh(glyphs, specular=1, specular_power=15,
#                  smooth_shading=True, show_scalar_bar=False)
# plotter.show()
#
#
#
#
#
#
#
# # mesh["scalars"] = n.ravel(order="F")
# # mesh["vectors"] = v.ravel(order="F")
# # print(mesh)
# #
# # # geom = pv.ParametricSuperEllipsoid(
# # #     xradius=1,
# # #     yradius=1,
# # #     zradius=1,
# # #     n1=0.2,
# # #     n2=0.4
# # #     )
# # #
# # # # make cool swirly pattern
# # # vectors = np.vstack(
# # #     (
# # #         np.sin(sphere.points[:, 0]),
# # #         np.cos(sphere.points[:, 1]),
# # #         np.cos(sphere.points[:, 2]),
# # #     )
# # # ).T
# # #
# # # # add and scale
# # # sphere.vectors = vectors * 0.3
# # #
# # # # plot just the arrows
# # # sphere.arrows.plot(scalars='GlyphScale')
# # #
# #
# #
# #
# # # mesh = examples.download_carotid().threshold(145, scalars="scalars")
# #
# # # Make a geometric object to use as the glyph
# # # geom = pv.Arrow()  # This could be any dataset
# #
# # # Perform the glyph
# # glyphs = mesh.glyph(orient="vectors", scale="scalars", factor=0.005, geom=geom)
# #
# # # plot using the plotting class
# # p = pv.Plotter()
# # p.add_mesh(glyphs)
# # # Set a cool camera position
# # p.camera_position = [
# #     (84.58052237950857, 77.76332116787425, 27.208569926456548),
# #     (131.39486171068918, 99.871379394528, 20.082859824932008),
# #     (0.13483731007732908, 0.033663777790747404, 0.9902957385932576),
# # ]
# #
# # p.show()
"""
