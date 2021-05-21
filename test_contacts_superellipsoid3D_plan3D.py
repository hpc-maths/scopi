import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyvista as pv
import scipy
from scipy.optimize import fsolve

class Plan3D:

    def __init__(self, x,y,z, theta=0, w=[0,0,1]):
        """
        En 3D : par defaut le plan est défini par un point et la normale (1,0,0)
        """
        # position d'un point du plan
        self.xc = x
        self.yc = y
        self.zc = z
        # normal definissant le plan : n = (0,0,1)
        # quaternion
        norm_w = np.linalg.norm(w)
        w = w/norm_w
        self.q = np.array([
            np.cos(theta/2),
            w[0]*np.sin(theta/2),
            w[1]*np.sin(theta/2),
            w[2]*np.sin(theta/2)])
        # print("\nq = \n",self.q)

    ## plan infini pas de scaling
    # def Ms(self):
    #     """
    #     compute scaling matrix
    #     """
    #     Ms = np.array([[self.rx,0,0,0],[0,self.ry,0,0],[0,0,self.rz,0],[0,0,0,1]])
    #     # print("\nMs = \n",Ms)
    #     return Ms

    def Mt(self):
        """
        compute translation matrix
        """
        Mt = np.array([[1,0,0,self.xc],[0,1,0,self.yc],[0,0,1,self.zc],[0,0,0,1]])
        # print("\nMt = \n",Mt)
        return Mt

    def Mr(self):
        """
        compute rotation matrix from quaternion
        """
        q0,q1,q2,q3 = self.q
        Mr = np.array([[ q0**2+q1**2-q2**2-q3**2, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2, 0 ],
                       [ 2*q1*q2+2*q0*q3, q0**2-q1**2+q2**2-q3**2, 2*q2*q3-2*q0*q1, 0 ],
                       [ 2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, q0**2-q1**2-q2**2+q3**2, 0 ],
                       [ 0,0,0,1]])
        # print("\nMr = \n",Mr)
        return Mr

    def M(self):
        M = self.Mt()@self.Mr()#@self.Ms()
        # print("\nM = \n",M)
        return M

    def surface_pt(self,a,b):
        """
        parametrage a l'aide de 2 vecteurs directeurs : (xc,yc,zc)+a*u+b*v où u et v sont vecteurs directeurs
        """
        q0,q1,q2,q3 = self.q
        ## vecteurs directeurs : (0,1,0) et (0,0,1)  => pt : a*(0,1,0)+b*(0,0,1)
        x = 0
        y = a
        z = b
        # On fait M@[x,y,z,1]
        pts = np.array([
            x*(q0**2 + q1**2 - q2**2 - q3**2) +
            y*(-2*q0*q3 + 2*q1*q2) +
            z*(2*q0*q2 + 2*q1*q3) + self.xc,
            x*(2*q0*q3 + 2*q1*q2) +
            y*(q0**2 - q1**2 + q2**2 - q3**2) +
            z*(-2*q0*q1 + 2*q2*q3) + self.yc,
            x*(-2*q0*q2 + 2*q1*q3) +
            y*(2*q0*q1 + 2*q2*q3) +
            z*(q0**2 - q1**2 - q2**2 + q3**2) + self.zc
            ]).T
        # print("\npoints : repere du solide = \n",np.array([x,y,z]).T)
        # print("\npoints : repere global = \n",pts)
        return pts

    def surface_normal(self,a,b,sign_p2s):
        """
        n = (1,0,0)
        """
        q0,q1,q2,q3 = self.q
        try:
            nx = sign_p2s*np.ones_like(a)
            ny = sign_p2s*np.zeros_like(a)
            nz = sign_p2s*np.zeros_like(a)
        except:
            nx = 1
            ny = 0
            nz = 0
        ## On applique la rotation
        normals = np.array([
            nx*(q0**2 + q1**2 - q2**2 - q3**2) +
            ny*(-2*q0*q3 + 2*q1*q2) +
            nz*(2*q0*q2 + 2*q1*q3),
            nx*(2*q0*q3 + 2*q1*q2) +
            ny*(q0**2 - q1**2 + q2**2 - q3**2) +
            nz*(-2*q0*q1 + 2*q2*q3),
            nx*(-2*q0*q2 + 2*q1*q3) +
            ny*(2*q0*q1 + 2*q2*q3) +
            nz*(q0**2 - q1**2 - q2**2 + q3**2)
            ]).T
        nn =  np.linalg.norm(normals,axis=1)
        normals[:,0] = normals[:,0]/nn
        normals[:,1] = normals[:,1]/nn
        normals[:,2] = normals[:,2]/nn
        # print("\nnormal : repere du solide = \n",np.array([nx,ny,nz]).T)
        # print("\nnormal : repere global = \n",normals)
        return normals

    def surface_tangent(self,a,b):
        """
        (0,1,0) et (0,0,1)
        """
        q0,q1,q2,q3 = self.q
        try:
            tgtx1 = np.zeros_like(a)
            tgty1 = np.ones_like(a)
            tgtz1 = np.zeros_like(a)
        except:
            tgtx1 = 0
            tgty1 = 1
            tgtz1 = 0
        ## On applique la rotation
        tangent1 = np.array([
            tgtx1*(q0**2 + q1**2 - q2**2 - q3**2) +
            tgty1*(-2*q0*q3 + 2*q1*q2) +
            tgtz1*(2*q0*q2 + 2*q1*q3),
            tgtx1*(2*q0*q3 + 2*q1*q2) +
            tgty1*(q0**2 - q1**2 + q2**2 - q3**2) +
            tgtz1*(-2*q0*q1 + 2*q2*q3),
            tgtx1*(-2*q0*q2 + 2*q1*q3) +
            tgty1*(2*q0*q1 + 2*q2*q3) +
            tgtz1*(q0**2 - q1**2 - q2**2 + q3**2)
            ]).T
        nn1 =  np.linalg.norm(tangent1,axis=1)
        tangent1[:,0] = tangent1[:,0]/nn1
        tangent1[:,1] = tangent1[:,1]/nn1
        tangent1[:,2] = tangent1[:,2]/nn1
        try:
            tgtx2 = np.zeros_like(a)
            tgty2 = np.zeros_like(a)
            tgtz2 = np.ones_like(a)
        except:
            tgtx2 = 0
            tgty2 = 0
            tgtz2 = 1
        ## On applique la rotation
        tangent2 = np.array([
            tgtx2*(q0**2 + q1**2 - q2**2 - q3**2) +
            tgty2*(-2*q0*q3 + 2*q1*q2) +
            tgtz2*(2*q0*q2 + 2*q1*q3),
            tgtx2*(2*q0*q3 + 2*q1*q2) +
            tgty2*(q0**2 - q1**2 + q2**2 - q3**2) +
            tgtz2*(-2*q0*q1 + 2*q2*q3),
            tgtx2*(-2*q0*q2 + 2*q1*q3) +
            tgty2*(2*q0*q1 + 2*q2*q3) +
            tgtz2*(q0**2 - q1**2 - q2**2 + q3**2)
            ]).T
        nn2 =  np.linalg.norm(tangent2,axis=1)
        tangent2[:,0] = tangent2[:,0]/nn2
        tangent2[:,1] = tangent2[:,1]/nn2
        tangent2[:,2] = tangent2[:,2]/nn2
        return tangent1,tangent2

    def mesh(self):
        mesh = pv.Plane(
            center=(0,0,0),
            direction=(1,0,0),
            i_size=30.0,
            j_size=30.0
            )
        mesh.transform(self.M())
        return mesh


class SuperEllipsoid3D:

    def __init__(self, x,y,z, rx,ry,rz, n=1, e=1, theta=0, w=[0,0,1]):
        """
        En 3D :   En 3D : ( |x/rx|**(2/e) + |y/ry|**(2/e) )**(e/n) + |z/rz|^(2/n) = 1
        Rotation d'angle theta autour du vecteur w : on utilise un quaternion
        0 < n <= 1 et 0 < e <= 1
        """
        # position
        self.xc = x
        self.yc = y
        self.zc = z
        # radius
        self.rx = rx
        self.ry = ry
        self.rz = rz
        # shape parameters
        self.e = e  # The “squareness” parameter in the x-y plane
        self.n = n  # The “squareness” parameter in the z plane
        # quaternion
        norm_w = np.linalg.norm(w)
        w = w/norm_w
        self.q = np.array([
            np.cos(theta/2),
            w[0]*np.sin(theta/2),
            w[1]*np.sin(theta/2),
            w[2]*np.sin(theta/2)])
        # print("\nq = \n",self.q)

    def Ms(self):
        """
        compute scaling matrix
        """
        Ms = np.array([[self.rx,0,0,0],[0,self.ry,0,0],[0,0,self.rz,0],[0,0,0,1]])
        # print("\nMs = \n",Ms)
        return Ms

    def Mt(self):
        """
        compute translation matrix
        """
        Mt = np.array([[1,0,0,self.xc],[0,1,0,self.yc],[0,0,1,self.zc],[0,0,0,1]])
        # print("\nMt = \n",Mt)
        return Mt

    def Mr(self):
        """
        compute rotation matrix from quaternion
        """
        q0,q1,q2,q3 = self.q
        Mr = np.array([[ q0**2+q1**2-q2**2-q3**2, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2, 0 ],
                       [ 2*q1*q2+2*q0*q3, q0**2-q1**2+q2**2-q3**2, 2*q2*q3-2*q0*q1, 0 ],
                       [ 2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, q0**2-q1**2-q2**2+q3**2, 0 ],
                       [ 0,0,0,1]])
        # print("\nMr = \n",Mr)
        return Mr

    def M(self):
        M = self.Mt()@self.Mr()@self.Ms()
        # print("\nM = \n",M)
        return M

    def surface_pt(self,a,b):
        """
        -pi/2 < a < pi/2 et -pi < b < pi
        """
        q0,q1,q2,q3 = self.q
        x = np.sign(np.cos(a))*np.abs(np.cos(a))**self.n * np.sign(np.cos(b))*np.abs(np.cos(b))**self.e
        y = np.sign(np.cos(a))*np.abs(np.cos(a))**self.n * np.sign(np.sin(b))*np.abs(np.sin(b))**self.e
        z = np.sign(np.sin(a))*np.abs(np.sin(a))**self.n
        # On fait M@[x,y,z,1]
        pts = np.array([
            self.rx*x*(q0**2 + q1**2 - q2**2 - q3**2) +
            self.ry*y*(-2*q0*q3 + 2*q1*q2) +
            self.rz*z*(2*q0*q2 + 2*q1*q3) + self.xc,
            self.rx*x*(2*q0*q3 + 2*q1*q2) +
            self.ry*y*(q0**2 - q1**2 + q2**2 - q3**2) +
            self.rz*z*(-2*q0*q1 + 2*q2*q3) + self.yc,
            self.rx*x*(-2*q0*q2 + 2*q1*q3) +
            self.ry*y*(2*q0*q1 + 2*q2*q3) +
            self.rz*z*(q0**2 - q1**2 - q2**2 + q3**2) + self.zc
            ]).T
        # print("\npoints : repere du solide = \n",np.array([x,y,z]).T)
        # print("\npoints : repere global = \n",pts)
        return pts

    def surface_normal(self,a,b):
        """
        -pi/2 < a < pi/2 et -pi < b < pi
        """
        q0,q1,q2,q3 = self.q
        nx = np.array( self.ry*self.rz*np.abs(np.cos(a))**(2-self.n)*np.sign(np.cos(b))*np.abs(np.cos(b))**(2-self.e) )
        ny = np.array( self.rx*self.rz*np.abs(np.cos(a))**(2-self.n)*np.sign(np.sin(b))*np.abs(np.sin(b))**(2-self.e) )
        nz = np.array( self.rx*self.ry*np.sign(np.cos(a))*np.sign(np.sin(a))*np.abs(np.sin(a))**(2-self.n) )
        ## On applique la rotation
        normals = np.array([
            nx*(q0**2 + q1**2 - q2**2 - q3**2) +
            ny*(-2*q0*q3 + 2*q1*q2) +
            nz*(2*q0*q2 + 2*q1*q3),
            nx*(2*q0*q3 + 2*q1*q2) +
            ny*(q0**2 - q1**2 + q2**2 - q3**2) +
            nz*(-2*q0*q1 + 2*q2*q3),
            nx*(-2*q0*q2 + 2*q1*q3) +
            ny*(2*q0*q1 + 2*q2*q3) +
            nz*(q0**2 - q1**2 - q2**2 + q3**2)
            ]).T
        nn =  np.linalg.norm(normals,axis=1)
        normals[:,0] = normals[:,0]/nn
        normals[:,1] = normals[:,1]/nn
        normals[:,2] = normals[:,2]/nn
        # print("\nnormal : repere du solide = \n",np.array([nx,ny,nz]).T)
        # print("\nnormal : repere global = \n",normals)
        return normals

    def surface_tangent(self,a,b):
        """
        -pi/2 < a < pi/2 et -pi < b < pi
        """
        q0,q1,q2,q3 = self.q
        # tgtx1 = -self.rx*self.n*np.sin(a)*np.abs(np.cos(a))**(self.n-1)*np.sign(np.cos(b))*np.abs(np.cos(b))**self.e
        # tgty1 = -self.ry*self.n*np.sin(a)*np.abs(np.cos(a))**(self.n-1)*np.sign(np.sin(b))*np.abs(np.sin(b))**self.e
        # tgtz1 = self.rz*self.n*np.cos(a)*np.abs(np.sin(a))**(self.n-1)
        tgtx1 = -self.rx*np.sign(np.cos(a))*np.sign(np.sin(b))*np.abs(np.sin(b))**(2-self.e)
        tgty1 =  self.ry*np.sign(np.cos(a))*np.sign(np.cos(b))*np.abs(np.cos(b))**(2-self.e)
        tgtz1 = np.zeros((b.shape[0]))
        ## On applique la rotation
        tangent1 = np.array([
            tgtx1*(q0**2 + q1**2 - q2**2 - q3**2) +
            tgty1*(-2*q0*q3 + 2*q1*q2) +
            tgtz1*(2*q0*q2 + 2*q1*q3),
            tgtx1*(2*q0*q3 + 2*q1*q2) +
            tgty1*(q0**2 - q1**2 + q2**2 - q3**2) +
            tgtz1*(-2*q0*q1 + 2*q2*q3),
            tgtx1*(-2*q0*q2 + 2*q1*q3) +
            tgty1*(2*q0*q1 + 2*q2*q3) +
            tgtz1*(q0**2 - q1**2 - q2**2 + q3**2)
            ]).T
        nn1 =  np.linalg.norm(tangent1,axis=1)
        tangent1[:,0] = tangent1[:,0]/nn1
        tangent1[:,1] = tangent1[:,1]/nn1
        tangent1[:,2] = tangent1[:,2]/nn1
        #
        # tgtx2 = -self.rx*self.e*np.sign(np.cos(a))*np.abs(np.cos(a))**self.n*np.abs(np.cos(b))**(self.e-1)*np.sin(b)
        # tgty2 =  self.ry*self.e*np.sign(np.cos(a))*np.abs(np.cos(a))**self.n*np.abs(np.sin(b))**(self.e-1)*np.cos(b)
        # tgtz2 = np.zeros((b.shape[0]))
        tgtx2 = -self.rx*np.sign(np.sin(a))*np.abs(np.sin(a))**(2-self.n)*np.sign(np.cos(b))*np.abs(np.cos(b))**self.e
        tgty2 = -self.ry*np.sign(np.sin(a))*np.abs(np.sin(a))**(2-self.n)*np.sign(np.sin(b))*np.abs(np.sin(b))**self.e
        tgtz2 =  self.rz*np.sign(np.cos(a))*np.abs(np.cos(a))**(2-self.n)
        ## On applique la rotation
        tangent2 = np.array([
            tgtx2*(q0**2 + q1**2 - q2**2 - q3**2) +
            tgty2*(-2*q0*q3 + 2*q1*q2) +
            tgtz2*(2*q0*q2 + 2*q1*q3),
            tgtx2*(2*q0*q3 + 2*q1*q2) +
            tgty2*(q0**2 - q1**2 + q2**2 - q3**2) +
            tgtz2*(-2*q0*q1 + 2*q2*q3),
            tgtx2*(-2*q0*q2 + 2*q1*q3) +
            tgty2*(2*q0*q1 + 2*q2*q3) +
            tgtz2*(q0**2 - q1**2 - q2**2 + q3**2)
            ]).T
        nn2 =  np.linalg.norm(tangent2,axis=1)
        tangent2[:,0] = tangent2[:,0]/nn2
        tangent2[:,1] = tangent2[:,1]/nn2
        tangent2[:,2] = tangent2[:,2]/nn2
        return tangent1,tangent2

    def mesh(self):
        mesh = pv.ParametricSuperEllipsoid(
            xradius=1,
            yradius=1,
            zradius=1,
            n1=self.n,
            n2=self.e
            )
        mesh.transform(self.M())
        return mesh

    def calcul_formel(self):
        import sympy
        from sympy.functions import sign, cos, sin, sqrt
        ## Pour la methode de newton
        a1,b1,a2,b2 = sympy.symbols("a1,b1,a2,b2",real=True)
        # superellipsoid1
        s1e = sympy.symbols("s1e",real=True)
        s1n = sympy.symbols("s1n",real=True)
        M00, M01, M02, M10, M11, M12, M20, M21, M22 = sympy.symbols("M00, M01, M02, M10, M11, M12, M20, M21, M22",real=True)
        s1Mr = sympy.Matrix([[ M00, M01, M02, 0 ],
                             [ M10, M11, M12, 0 ],
                             [ M20, M21, M22, 0 ],
                             [ 0,0,0,1]])
        s1rx,s1ry,s1rz,s1xc,s1yc,s1zc = sympy.symbols("s1rx,s1ry,s1rz,s1xc,s1yc,s1zc",real=True)
        s1Ms= sympy.Matrix([[s1rx,0,0,0],[0,s1ry,0,0],[0,0,s1rz,0],[0,0,0,1]])
        s1Mt = sympy.Matrix([[1,0,0,s1xc],[0,1,0,s1yc],[0,0,1,s1zc],[0,0,0,1]])
        s1M = s1Mt@s1Mr@s1Ms
        s1x = sign(cos(a1))*sympy.Abs(cos(a1))**s1n*sign(cos(b1))*sympy.Abs(cos(b1))**s1e
        s1y = sign(cos(a1))*sympy.Abs(cos(a1))**s1n*sign(sin(b1))*sympy.Abs(sin(b1))**s1e
        s1z = sign(sin(a1))*sympy.Abs(sin(a1))**s1n
        pt1 = s1M@sympy.Matrix([s1x,s1y,s1z,1])
        s1nx = s1ry*s1rz*sympy.Abs(cos(a1))**(2-s1n)*sign(cos(b1))*sympy.Abs(cos(b1))**(2-s1e)
        s1ny = s1rx*s1rz*sympy.Abs(cos(a1))**(2-s1n)*sign(sin(b1))*sympy.Abs(sin(b1))**(2-s1e)
        s1nz = s1rx*s1ry*sign(cos(a1))*sign(sin(a1))*sympy.Abs(sin(a1))**(2-s1n)
        n1 = s1Mr@sympy.Matrix([s1nx,s1ny,s1nz,1])
        print("\ncalcul formel : pt1 = \n",pt1)
        print("\ncalcul formel : n1 = \n",n1)

        # plan
        N00, N01, N02, N10, N11, N12, N20, N21, N22 = sympy.symbols("N00, N01, N02, N10, N11, N12, N20, N21, N22",real=True)
        p2Mr = sympy.Matrix([[ N00, N01, N02, 0 ],
                            [ N10, N11, N12, 0 ],
                            [ N20, N21, N22, 0 ],
                            [ 0,0,0,1]])
        p2xc,p2yc,p2zc = sympy.symbols("p2xc,p2yc,p2zc",real=True)
        p2Mt = sympy.Matrix([[1,0,0,p2xc],[0,1,0,p2yc],[0,0,1,p2zc],[0,0,0,1]])
        p2M = p2Mt@p2Mr
        p2x = 0
        p2y = a2
        p2z = b2
        pt2 = p2M@sympy.Matrix([p2x,p2y,p2z,1])
        sign_p2s = sympy.symbols("sign_p2s",real=True)
        p2nx = sign_p2s
        p2ny = 0
        p2nz = 0
        n2 = p2Mr@sympy.Matrix([p2nx,p2ny,p2nz,1])
        print("\ncalcul formel : pt2 = \n",pt2)
        print("\ncalcul formel : n2 = \n",n2)

        # (y2-y1)*nz1 - (z2-z1)*ny1 = 0
        fct0 =  (pt2[1]-pt1[1])*n1[2] - (pt2[2]-pt1[2])*n1[1]
        print("\ncalcul formel : fct0 = \n",fct0)
        # (z2-z1)*nx1 - (x2-x1)*nz1 = 0
        fct1 =  (pt2[2]-pt1[2])*n1[0] - (pt2[0]-pt1[0])*n1[2]
        print("\ncalcul formel : fct1 = \n",fct1)
        # (x2-x1)*ny1 - (y2-y1)*nx1 = 0
        fct2 =  (pt2[0]-pt1[0])*n1[1] - (pt2[1]-pt1[1])*n1[0]
        print("\ncalcul formel : fct2 = \n",fct2)
        # n1.n2 + ||n1||*||n2|| = 0 (i.e. n1=-n2)
        # fct3 = sympy.simplify( n1[0]*n2[0]+n1[1]*n2[1] )# en considerant n normalisé... +sqrt(n1[0]*n1[0]+n1[1]*n1[1])+sqrt(n2[0]*n2[0]+n2[1]*n2[1])
        fct3 = n1[0]*n2[0]+n1[1]*n2[1]+n1[2]*n2[2] + sqrt(n1[0]*n1[0]+n1[1]*n1[1]+n1[2]*n1[2])*sqrt(n2[0]*n2[0]+n2[1]*n2[1]+n2[2]*n2[2])
        print("\ncalcul formel : fct3 = \n",fct3)

        dfct0da1 = sympy.diff(fct0 ,a1)
        dfct0db1 = sympy.diff(fct0 ,b1)
        dfct0da2 = sympy.diff(fct0 ,a2)
        dfct0db2 = sympy.diff(fct0 ,b2)

        dfct1da1 = sympy.diff(fct1 ,a1)
        dfct1db1 = sympy.diff(fct1 ,b1)
        dfct1da2 = sympy.diff(fct1 ,a2)
        dfct1db2 = sympy.diff(fct1 ,b2)

        dfct2da1 = sympy.diff(fct2 ,a1)
        dfct2db1 = sympy.diff(fct2 ,b1)
        dfct2da2 = sympy.diff(fct2 ,a2)
        dfct2db2 = sympy.diff(fct2 ,b2)

        dfct3da1 = sympy.diff(fct3 ,a1)
        dfct3db1 = sympy.diff(fct3 ,b1)
        dfct3da2 = sympy.diff(fct3 ,a2)
        dfct3db2 = sympy.diff(fct3 ,b2)

        print("\ncalcul formel : dfct0da1 = \n",dfct0da1)
        print("\ncalcul formel : dfct0db1 = \n",dfct0db1)
        print("\ncalcul formel : dfct0da2 = \n",dfct0da2)
        print("\ncalcul formel : dfct0db2 = \n",dfct0db2)

        print("\ncalcul formel : dfct1da1 = \n",dfct1da1)
        print("\ncalcul formel : dfct1db1 = \n",dfct1db1)
        print("\ncalcul formel : dfct1da2 = \n",dfct1da2)
        print("\ncalcul formel : dfct1db2 = \n",dfct1db2)

        print("\ncalcul formel : dfct2da1 = \n",dfct2da1)
        print("\ncalcul formel : dfct2db1 = \n",dfct2db1)
        print("\ncalcul formel : dfct2da2 = \n",dfct2da2)
        print("\ncalcul formel : dfct2db2 = \n",dfct2db2)

        print("\ncalcul formel : dfct3da1 = \n",dfct3da1)
        print("\ncalcul formel : dfct3db1 = \n",dfct3db1)
        print("\ncalcul formel : dfct3da2 = \n",dfct3da2)
        print("\ncalcul formel : dfct3db2 = \n",dfct3db2)

if __name__ == '__main__':

    s1 = SuperEllipsoid3D(0,0,0, 2,1,1, n=0.9, e=0.8, theta=65, w=[0,-1,1])
    p2 = Plan3D(6,0,0, theta=45, w=[0,0,1])

    sign_p2s = np.sign(np.dot( np.array([s1.xc-p2.xc,s1.yc-p2.yc,s1.zc-p2.zc]), p2.surface_normal([0],[0],1)[0] ))
    print("sign_p2s =",sign_p2s)

    s1.calcul_formel()
    # sys.exit()

    p = pv.Plotter()
    grid = pv.UniformGrid()

    L = 10
    arr = np.arange((2*L)**3).reshape((2*L,2*L,2*L))
    grid.dimensions = np.array(arr.shape) + 1 #dim + 1 because cells
    grid.origin = (-L, -L, -L)
    grid.spacing = (1, 1, 1)
    p.add_mesh(grid, show_edges=True, opacity=0.1)

    p.add_mesh(s1.mesh(),color="red", opacity=0.5)
    p.add_mesh(p2.mesh(),color="green", opacity=0.5)

    ## Qq pts sur la superellipsoide
    aa = np.linspace(-np.pi/2,np.pi/2,num=20)
    bb = np.linspace(-np.pi,np.pi,num=20)
    aaaa, bbbb = np.meshgrid(aa,bb)
    aa = aaaa.reshape((-1,))
    bb = bbbb.reshape((-1,))
    pts1 = s1.surface_pt(aa,bb)
    normals1 = s1.surface_normal(aa,bb)
    tgt11,tgt12 = s1.surface_tangent(aa,bb)
    nodes1 = pv.PolyData(pts1)
    nodes1["normal"] = normals1
    nodes1["tangent1"] = tgt11
    nodes1["tangent2"] = tgt12
    p.add_mesh(nodes1.glyph(factor=0.05, geom=pv.Sphere()),color="blue")
    p.add_mesh(nodes1.glyph(orient="normal",factor=0.25, geom=pv.Arrow()),color="blue")
    p.add_mesh(nodes1.glyph(orient="tangent1",factor=0.25, geom=pv.Arrow()),color="magenta")
    p.add_mesh(nodes1.glyph(orient="tangent2",factor=0.25, geom=pv.Arrow()),color="magenta")

    ## Qq pts sur le plan
    aa = np.linspace(-20,20,num=20)
    bb = np.linspace(-20,20,num=20)
    aaaa, bbbb = np.meshgrid(aa,bb)
    aa = aaaa.reshape((-1,))
    bb = bbbb.reshape((-1,))
    pts2 = p2.surface_pt(aa,bb)
    normals2 = p2.surface_normal(aa,bb,sign_p2s)
    tgt21,tgt22 = p2.surface_tangent(aa,bb)
    nodes2 = pv.PolyData(pts2)
    nodes2["normal"] = normals2
    nodes2["tangent1"] = tgt21
    nodes2["tangent2"] = tgt22
    p.add_mesh(nodes2.glyph(factor=0.05, geom=pv.Sphere()),color="blue")
    p.add_mesh(nodes2.glyph(orient="normal",factor=0.25, geom=pv.Arrow()),color="blue")
    p.add_mesh(nodes2.glyph(orient="tangent1",factor=0.25, geom=pv.Arrow()),color="magenta")
    p.add_mesh(nodes2.glyph(orient="tangent2",factor=0.25, geom=pv.Arrow()),color="magenta")

    # Calcul de la distance entre une superellipsoide et un plan
    # (1) appartenance à la superellipsoide  => a1,b1 => x1=x(a1,b1), y1=y(a1,b1)
    # (2) appartenance au plan => a2,b2 => x2=x(a2,b2), y2=y(a2,b2)
    # (3) soient X1=(x1,y1) X2=(x2,y2)
    #     on veut X2X1 ^ n1 = 0 (i.e. X2X1 et n1 colineaires)
    #     (x2-x1)   nx1         (y2-y1)*nz1 - (z2-z1)*ny1 = 0
    #     (y2-y1)   ny1    i.e. (z2-z1)*nx1 - (x2-x1)*nz1 = 0
    #     (z2-z1)   nz1         (x2-x1)*ny1 - (y2-y1)*nx1 = 0
    #     (x2-x1)   nx1
    #     (y2-y1)   ny1
    # (4) normales opposées : n1.n2 + ||n1||*||n2|| = 0 (i.e. n1=-n2)

    def f_contacts(u,s1,p2,sign_p2s):
        a1,b1,a2,b2 = u
        res = np.zeros((4,))
        s1q0,s1q1,s1q2,s1q3 = s1.q
        s1n = s1.n ; s1e = s1.e ; s1rx = s1.rx ; s1ry = s1.ry ; s1rz = s1.rz ; s1xc = s1.xc ; s1yc = s1.yc ; s1zc = s1.zc
        p2q0,p2q1,p2q2,p2q3 = p2.q
        p2xc = p2.xc ; p2yc = p2.yc ; p2zc = p2.zc

        s1q00 = s1q0**2
        s1q11 = s1q1**2
        s1q22 = s1q2**2
        s1q33 = s1q3**2
        s1q03 = 2*s1q0*s1q3
        s1q12 = 2*s1q1*s1q2
        M00 = (s1q00 + s1q11 - s1q22 - s1q33)  # M1[0,0]
        M01 = (-s1q03 + s1q12) # M1[0,1]
        M02 = (2*s1q0*s1q2 + 2*s1q1*s1q3)    # M1[0,2]
        M10 = (s1q03 + s1q12)  # M1[1,0]
        M11 = (s1q00 - s1q11 + s1q22 - s1q33)  # M1[1,1]
        M12 = (-2*s1q0*s1q1 + 2*s1q2*s1q3)  # M1[1,2]
        M20 = (-2*s1q0*s1q2 + 2*s1q1*s1q3)  # M1[2,0]
        M21 = (2*s1q0*s1q1 + 2*s1q2*s1q3)   # M1[2,1]
        M22 = (s1q00 - s1q11 - s1q22 + s1q33)   # M1[2,2]

        p2q00 = p2q0**2
        p2q11 = p2q1**2
        p2q22 = p2q2**2
        p2q33 = p2q3**2
        p2q03 = 2*p2q0*p2q3
        p2q12 = 2*p2q1*p2q2
        N00 = (p2q00 + p2q11 - p2q22 - p2q33)  # M2[0,0]
        N01 = (-p2q03 + p2q12) # M2[0,1]
        N02 = (2*p2q0*p2q2 + 2*p2q1*p2q3)    # M2[0,2]
        N10 = (p2q03 + p2q12)  # M2[1,0]
        N11 = (p2q00 - p2q11 + p2q22 - p2q33)  # M2[1,1]
        N12 = (-2*p2q0*p2q1 + 2*p2q2*p2q3)  # M2[1,2]
        N20 = (-2*p2q0*p2q2 + 2*p2q1*p2q3)  # M2[2,0]
        N21 = (2*p2q0*p2q1 + 2*p2q2*p2q3)   # M2[2,1]
        N22 = (p2q00 - p2q11 - p2q22 + p2q33)   # M2[2,2]
        #
        ca1 = np.cos(a1)
        ca1n = np.sign(ca1)*np.abs(ca1)**s1n
        ca1n2 = np.abs(ca1)**(2 - s1n)
        #
        sa1 = np.sin(a1)
        sa1n = np.abs(sa1)**s1n*np.sign(sa1)
        sa1n2 = np.abs(sa1)**(2 - s1n)*np.sign(sa1) # ou np.sign(sa1)*np.abs(sa1)**(2 - s1n)

        #
        cb1 = np.cos(b1)
        cb1e = np.sign(cb1)*np.abs(cb1)**s1e # ou np.abs(cb1)**s1e*np.sign(cb1)
        cb1e2 = np.abs(cb1)**(2 - s1e)*np.sign(cb1)# ou np.sign(cb1)*np.abs(cb1)**(2 - s1e)
        #
        sb1 = np.sin(b1)
        sb1e = np.abs(sb1)**s1e*np.sign(sb1)
        sb1e2 = np.abs(sb1)**(2 - s1e)*np.sign(sb1) # ou np.sign(sb1)*np.abs(sb1)**(2 - s1e)
        #
        A1 = s1ry * s1rz * ca1n2 * cb1e2
        A2 = s1rx * s1rz * sb1e2 * ca1n2
        A3 = s1rx * s1ry * sa1n2 * np.sign(ca1)
        A4 = s1rx * ca1n  * cb1e
        A5 = s1ry * sb1e  * ca1n
        A6 = s1rz * sa1n

        res = np.zeros((4,))

        res[0] = - (  M10*A1 + M11*A2 + M12*A3 ) \
                 * ( -M20*A4 - M21*A5 - M22*A6 + N21*a2 + N22*b2 + p2zc - s1zc ) \
                 + (  M20*A1 + M21*A2 + M22*A3 ) \
                 * ( -M10*A4 - M11*A5 - M12*A6 + N11*a2 + N12*b2 + p2yc - s1yc )

        res[1] =   (  M00*A1 + M01*A2 + M02*A3 ) \
                 * ( -M20*A4 - M21*A5 - M22*A6 + N21*a2 + N22*b2 + p2zc - s1zc ) \
                 - (  M20*A1 + M21*A2 + M22*A3 ) \
                 * ( -M00*A4 - M01*A5 - M02*A6 + N01*a2 + N02*b2 + p2xc - s1xc )

        res[2] = - (  M00*A1 + M01*A2 + M02*A3 ) \
                 * ( -M10*A4 - M11*A5 - M12*A6 + N11*a2 + N12*b2 + p2yc - s1yc ) \
                 + (  M10*A1 + M11*A2 + M12*A3 ) \
                 * ( -M00*A4 - M01*A5 - M02*A6 + N01*a2 + N02*b2 + p2xc - s1xc )

        res[3] =   N00*sign_p2s*( M00*A1 + M01*A2 + M02*A3 ) \
                 + N10*sign_p2s*( M10*A1 + M11*A2 + M12*A3 ) \
                 + N20*sign_p2s*( M20*A1 + M21*A2 + M22*A3 ) \
                 + np.sqrt( N00**2 + N10**2 + N20**2 ) \
                 * np.sqrt(  ( M00*A1 + M01*A2 + M02*A3 )**2 \
                           + ( M10*A1 + M11*A2 + M12*A3 )**2 \
                           + ( M20*A1 + M21*A2 + M22*A3 )**2 )


        ## res_ref est directement obtenu a partir du calcul formel
        ## permet de tester que la reecriture des formules n'a pas fait ajouter des erreurs
        res_ref = np.zeros((4,))

        res_ref[0] = -( M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) \
                      + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) \
                      + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) )\
                     *(-M20*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) \
                      - M21*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) \
                      - M22*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N21*a2 + N22*b2 + p2zc - s1zc ) \
                     +( M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) \
                      + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) \
                      + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) )\
                     *(-M10*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) \
                     - M11*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) \
                     - M12*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N11*a2 + N12*b2 + p2yc - s1yc)

        res_ref[1] = ( M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) \
                     + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) \
                     + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) ) \
                    *(-M20*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) \
                     - M21*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) \
                     - M22*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N21*a2 + N22*b2 + p2zc - s1zc )\
                    -( M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) \
                     + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) \
                     + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) ) \
                    *(-M00*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) \
                     - M01*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) \
                     - M02*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N01*a2 + N02*b2 + p2xc - s1xc)

        res_ref[2] = -( M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) \
                      + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) \
                      + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) )\
                     *(-M10*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) \
                      - M11*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) \
                      - M12*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N11*a2 + N12*b2 + p2yc - s1yc) \
                     +( M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) \
                      + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) \
                      + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1))) \
                     *(-M00*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) \
                      - M01*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) \
                      - M02*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N01*a2 + N02*b2 + p2xc - s1xc)

        res_ref[3] = N00*sign_p2s*( M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) \
                                  + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) \
                                  + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1))) \
                   + N10*sign_p2s*( M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) \
                                  + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) \
                                  + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1))) \
                   + N20*sign_p2s*( M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) \
                                  + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) \
                                  + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1))) \
                   + np.sqrt( N00**2*sign_p2s**2 + N10**2*sign_p2s**2 + N20**2*sign_p2s**2 )\
                    *np.sqrt( ( M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) \
                              + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) \
                              + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) )**2 \
                            + ( M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) \
                              + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) \
                              + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) )**2 \
                            + ( M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) \
                              + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) \
                              + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) )**2 )

        # print("F : res = ",res," res_ref = ",res_ref," ecart = ",np.linalg.norm(res-res_ref))

        return res

    def DiracDelta(x):
        return 0

    def grad_f_contacts(u,s1,s2,sign_p2s):
        a1,b1,a2,b2 = u
        res = np.zeros((4,4))
        s1q0,s1q1,s1q2,s1q3 = s1.q
        s1n = s1.n ; s1e = s1.e ; s1rx = s1.rx ; s1ry = s1.ry ; s1rz = s1.rz ; s1xc = s1.xc ; s1yc = s1.yc ; s1zc = s1.zc
        p2q0,p2q1,p2q2,p2q3 = p2.q
        p2xc = p2.xc ; p2yc = p2.yc ; p2zc = p2.zc

        s1q00 = s1q0**2
        s1q11 = s1q1**2
        s1q22 = s1q2**2
        s1q33 = s1q3**2
        s1q03 = 2*s1q0*s1q3
        s1q12 = 2*s1q1*s1q2
        M00 = (s1q00 + s1q11 - s1q22 - s1q33)  # M1[0,0]
        M01 = (-s1q03 + s1q12) # M1[0,1]
        M02 = (2*s1q0*s1q2 + 2*s1q1*s1q3)    # M1[0,2]
        M10 = (s1q03 + s1q12)  # M1[1,0]
        M11 = (s1q00 - s1q11 + s1q22 - s1q33)  # M1[1,1]
        M12 = (-2*s1q0*s1q1 + 2*s1q2*s1q3)  # M1[1,2]
        M20 = (-2*s1q0*s1q2 + 2*s1q1*s1q3)  # M1[2,0]
        M21 = (2*s1q0*s1q1 + 2*s1q2*s1q3)   # M1[2,1]
        M22 = (s1q00 - s1q11 - s1q22 + s1q33)   # M1[2,2]

        p2q00 = p2q0**2
        p2q11 = p2q1**2
        p2q22 = p2q2**2
        p2q33 = p2q3**2
        p2q03 = 2*p2q0*p2q3
        p2q12 = 2*p2q1*p2q2
        N00 = (p2q00 + p2q11 - p2q22 - p2q33)  # M2[0,0]
        N01 = (-p2q03 + p2q12) # M2[0,1]
        N02 = (2*p2q0*p2q2 + 2*p2q1*p2q3)    # M2[0,2]
        N10 = (p2q03 + p2q12)  # M2[1,0]
        N11 = (p2q00 - p2q11 + p2q22 - p2q33)  # M2[1,1]
        N12 = (-2*p2q0*p2q1 + 2*p2q2*p2q3)  # M2[1,2]
        N20 = (-2*p2q0*p2q2 + 2*p2q1*p2q3)  # M2[2,0]
        N21 = (2*p2q0*p2q1 + 2*p2q2*p2q3)   # M2[2,1]
        N22 = (p2q00 - p2q11 - p2q22 + p2q33)   # M2[2,2]
        #
        ca1 = np.cos(a1)
        ca1n = np.sign(ca1)*np.abs(ca1)**s1n # ou np.abs(ca1)**s1n*np.sign(ca1)
        ca1n1 = s1n*np.abs(ca1)**(s1n - 1)
        ca1n2 = np.abs(ca1)**(2 - s1n)
        ca1n3 = (2 - s1n)*np.sign(ca1)*np.abs(ca1)**(1 - s1n)

        #
        sa1 = np.sin(a1)
        sa1n = np.abs(sa1)**s1n*np.sign(sa1)
        sa1n1 = s1n*np.abs(sa1)**(s1n - 1)
        sa1n2 = np.abs(sa1)**(2 - s1n)*np.sign(sa1) # ou np.sign(sa1)*np.abs(sa1)**(2 - s1n)
        sa1n3 = (2 - s1n)*np.abs(sa1)**(1 - s1n)

        #
        cb1 = np.cos(b1)
        cb1e = np.sign(cb1)*np.abs(cb1)**s1e # ou np.abs(cb1)**s1e*np.sign(cb1)
        cb1e1 = s1e*np.abs(cb1)**(s1e - 1)
        cb1e2 = np.abs(cb1)**(2 - s1e)*np.sign(cb1)# ou np.sign(cb1)*np.abs(cb1)**(2 - s1e)
        cb1e3 = (2 - s1e)*np.abs(cb1)**(1 - s1e)
        #
        sb1 = np.sin(b1)
        sb1e = np.abs(sb1)**s1e*np.sign(sb1)
        sb1e1 = s1e*np.abs(sb1)**(s1e - 1)
        sb1e2 = np.abs(sb1)**(2 - s1e)*np.sign(sb1) # ou np.sign(sb1)*np.abs(sb1)**(2 - s1e)
        sb1e3 = (2 - s1e)*np.abs(sb1)**(1 - s1e)
        #

        A1 = s1ry * s1rz * ca1n2 * cb1e2
        A2 = s1rx * s1rz * sb1e2 * ca1n2
        A3 = s1rx * s1ry * sa1n2 * np.sign(ca1)
        A4 = s1rx * ca1n  * cb1e
        A5 = s1ry * sb1e  * ca1n
        A6 = s1rz * sa1n
        A7 = s1rx * sa1 * ca1n1 * cb1e
        A8 = s1ry * sa1 * sb1e  * ca1n1
        A9 = s1rz * ca1 * sa1n1
        A10 = s1ry * s1rz * sa1 * ca1n3 * cb1e2
        A11 = s1rx * s1rz * sa1 * sb1e2 * ca1n3
        A12 = s1rx * s1ry * np.abs(ca1) * sa1n3
        A13 = s1rx * sb1 * ca1n  * cb1e1
        A14 = s1ry * cb1 * sb1e1 * ca1n
        A15 = s1ry * s1rz  * sb1 * ca1n2 * cb1e3
        A16 = s1rx * s1rz  * cb1 * sb1e3 * ca1n2

        res[0,0] = ( -M10*A1 - M11*A2 - M12*A3 ) * (  M20*A7 + M21*A8 - M22*A9 ) \
                  +(  M20*A1 + M21*A2 + M22*A3 ) * (  M10*A7 + M11*A8 - M12*A9 ) \
                  +(  M10*A10 + M11*A11 - M12*A12 ) * ( -M20*A4 - M21*A5 - M22*A6 + N21*a2 + N22*b2 + p2zc - s1zc ) \
                  +( -M20*A10 - M21*A11 + M22*A12 ) * ( -M10*A4 - M11*A5 - M12*A6 + N11*a2 + N12*b2 + p2yc - s1yc )

        res[0,1] = ( -M10*A1 - M11*A2 - M12*A3 ) * ( M20*A13 - M21*A14 ) \
                  +(  M20*A1 + M21*A2 + M22*A3 ) * ( M10*A13 - M11*A14 ) \
                  +(  M10*A15 - M11*A16 ) * ( -M20*A4 - M21*A5 - M22*A6 + N21*a2 + N22*b2 + p2zc - s1zc ) \
                  +( -M20*A15 + M21*A16 ) * ( -M10*A4 - M11*A5 - M12*A6 + N11*a2 + N12*b2 + p2yc - s1yc )

        res[0,2] = N11*( M20*A1 + M21*A2 + M22*A3 ) + N21*( -M10*A1 - M11*A2 - M12*A3 )

        res[0,3] = N12*( M20*A1 + M21*A2 + M22*A3 ) + N22*( -M10*A1 - M11*A2 - M12*A3 )

        res[1,0] = (  M00*A1 + M01*A2 + M02*A3 ) * ( M20*A7 + M21*A8 - M22*A9 ) \
                  +( -M20*A1 - M21*A2 - M22*A3 ) * ( M00*A7 + M01*A8 - M02*A9 ) \
                  +( -M00*A10 - M01*A11 + M02*A12 ) * ( -M20*A4 - M21*A5 - M22*A6 + N21*a2 + N22*b2 + p2zc - s1zc ) \
                 + (  M20*A10 + M21*A11 - M22*A12 ) * ( -M00*A4 - M01*A5 - M02*A6 + N01*a2 + N02*b2 + p2xc - s1xc )

        res[1,1] = (  M00*A1 + M01*A2 + M02*A3 ) * ( M20*A13 - M21*A14 ) \
                  +( -M20*A1 - M21*A2 - M22*A3 ) * ( M00*A13 - M01*A14 ) \
                  +( -M00*A15 + M01*A16 ) * ( -M20*A4 - M21*A5 - M22*A6 + N21*a2 + N22*b2 + p2zc - s1zc ) \
                  +(  M20*A15 - M21*A16 ) * ( -M00*A4 - M01*A5 - M02*A6 + N01*a2 + N02*b2 + p2xc - s1xc )

        res[1,2] = N01*( -M20*A1 - M21*A2 - M22*A3 ) + N21*(  M00*A1 + M01*A2 + M02*A3 )

        res[1,3] = N02*(  -M20*A1 - M21*A2 - M22*A3 ) + N22*(   M00*A1 + M01*A2 + M02*A3 )

        res[2,0] = ( -M00*A1 - M01*A2 - M02*A3 ) * ( M10*A7 + M11*A8 - M12*A9 ) \
                  +(  M10*A1 + M11*A2 + M12*A3 ) * ( M00*A7 + M01*A8 - M02*A9 ) \
                  +(  M00*A10 + M01*A11 - M02*A12 ) * ( -M10*A4 - M11*A5 - M12*A6 + N11*a2 + N12*b2 + p2yc - s1yc ) \
                  +(  -M10*A10 - M11*A11 + M12*A12) *( -M00*A4 - M01*A5 - M02*A6 + N01*a2 + N02*b2 + p2xc - s1xc )

        res[2,1] = (-M00*A1 - M01*A2 - M02*A3 ) * ( M10*A13 - M11*A14 ) \
                 + ( M10*A1 + M11*A2 + M12*A3 ) * ( M00*A13 - M01*A14 ) \
                 + ( M00*A15 - M01*A16 ) * (-M10*A4 - M11*A5 - M12*A6 + N11*a2 + N12*b2 + p2yc - s1yc ) \
                  +(-M10*A15 + M11*A16 ) * (-M00*A4 - M01*A5 - M02*A6 + N01*a2 + N02*b2 + p2xc - s1xc )

        res[2,2] =  N01*( M10*A1 + M11*A2 + M12*A3 ) + N11*(-M00*A1 - M01*A2 - M02*A3 )

        res[2,3] = N02*(   M10*A1 + M11*A2 + M12*A3 ) + N12*(  -M00*A1 - M01*A2 - M02*A3 )

        res[3,0] = N00*sign_p2s*(-M00*A10 - M01*A11 + M02*A12 ) \
                 + N10*sign_p2s*(-M10*A10 - M11*A11 + M12*A12 ) \
                 + N20*sign_p2s*(-M20*A10 - M21*A11 + M22*A12 ) \
                 + np.sqrt( N00**2 + N10**2 + N20**2 ) \
                 * ( ( M00*A1 + M01*A2 + M02*A3 ) * (-M00*A10 - M01*A11 + M02*A12 ) \
                   + ( M10*A1 + M11*A2 + M12*A3 ) * (-M10*A10 - M11*A11 + M12*A12 ) \
                   + ( M20*A1 + M21*A2 + M22*A3 ) * (-M20*A10 - M21*A11 + M22*A12 ) ) \
                 / np.sqrt( ( M00*A1 + M01*A2 + M02*A3 )**2 \
                          + ( M10*A1 + M11*A2 + M12*A3 )**2 \
                          + ( M20*A1 + M21*A2 + M22*A3 )**2 )

        res[3,1] = N00*sign_p2s*(-M00*A15 + M01*A16 ) \
                 + N10*sign_p2s*(-M10*A15 + M11*A16 ) \
                 + N20*sign_p2s*(-M20*A15 + M21*A16 ) \
                 + np.sqrt( N00**2 + N10**2 + N20**2 )\
                 * ( ( M00*A1 + M01*A2 + M02*A3 ) * (-M00*A15 + M01*A16 ) \
                   + ( M10*A1 + M11*A2 + M12*A3 ) * (-M10*A15 + M11*A16 ) \
                   + ( M20*A1 + M21*A2 + M22*A3 ) * (-M20*A15 + M21*A16 ) ) \
                 / np.sqrt( ( M00*A1 + M01*A2 + M02*A3 )**2 \
                          + ( M10*A1 + M11*A2 + M12*A3 )**2 \
                          + ( M20*A1 + M21*A2 + M22*A3 )**2 )

        res[3,2] = 0

        res[3,3] = 0




















        ## res_ref est directement obtenu a partir du calcul formel
        ## permet de tester que la reecriture des formules n'a pas fait ajouter des erreurs
        res_ref = np.zeros((4,4))
        res_ref[0,0] = (-M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) - M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))*(M20*s1n*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + 2*M20*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1n*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) + 2*M21*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - M22*s1n*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) - 2*M22*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))*(M10*s1n*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + 2*M10*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1n*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) + 2*M11*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - M12*s1n*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) - 2*M12*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (M10*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + M11*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - M12*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) + 2*M12*s1rx*s1ry*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) - 2*M12*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)))*(-M20*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M21*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M22*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N21*a2 + N22*b2 + p2zc - s1zc) + (-M20*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)) - M21*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) + M22*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 2*M22*s1rx*s1ry*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 2*M22*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)))*(-M10*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M11*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M12*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N11*a2 + N12*b2 + p2yc - s1yc)
        res_ref[0,1] = (-M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) - M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))*(M20*s1e*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*M20*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - M21*s1e*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) - 2*M21*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))*(M10*s1e*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*M10*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - M11*s1e*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) - 2*M11*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (M10*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*M10*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)) - M11*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) - 2*M11*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)))*(-M20*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M21*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M22*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N21*a2 + N22*b2 + p2zc - s1zc) + (-M20*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*M20*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)) + M21*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*M21*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)))*(-M10*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M11*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M12*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N11*a2 + N12*b2 + p2yc - s1yc)
        res_ref[0,2] = N11*(M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1))) + N21*(-M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) - M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))
        res_ref[0,3] = N12*(M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1))) + N22*(-M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) - M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))
        res_ref[1,0] = (M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))*(M20*s1n*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + 2*M20*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1n*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) + 2*M21*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - M22*s1n*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) - 2*M22*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (-M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) - M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))*(M00*s1n*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + 2*M00*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1n*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) + 2*M01*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - M02*s1n*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) - 2*M02*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (-M00*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)) - M01*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) + M02*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 2*M02*s1rx*s1ry*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 2*M02*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)))*(-M20*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M21*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M22*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N21*a2 + N22*b2 + p2zc - s1zc) + (M20*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + M21*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - M22*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) + 2*M22*s1rx*s1ry*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) - 2*M22*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)))*(-M00*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M01*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M02*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N01*a2 + N02*b2 + p2xc - s1xc)
        res_ref[1,1] = (M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))*(M20*s1e*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*M20*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - M21*s1e*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) - 2*M21*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (-M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) - M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))*(M00*s1e*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*M00*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - M01*s1e*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) - 2*M01*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (-M00*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*M00*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)) + M01*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*M01*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)))*(-M20*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M21*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M22*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N21*a2 + N22*b2 + p2zc - s1zc) + (M20*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*M20*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)) - M21*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) - 2*M21*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)))*(-M00*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M01*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M02*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N01*a2 + N02*b2 + p2xc - s1xc)
        res_ref[1,2] = N01*(-M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) - M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1))) + N21*(M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))
        res_ref[1,3] = N02*(-M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) - M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1))) + N22*(M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))
        res_ref[2,0] = (-M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) - M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))*(M10*s1n*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + 2*M10*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1n*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) + 2*M11*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - M12*s1n*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) - 2*M12*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))*(M00*s1n*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + 2*M00*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1n*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) + 2*M01*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - M02*s1n*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) - 2*M02*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (M00*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + M01*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - M02*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) + 2*M02*s1rx*s1ry*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) - 2*M02*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)))*(-M10*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M11*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M12*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N11*a2 + N12*b2 + p2yc - s1yc) + (-M10*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)) - M11*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) + M12*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 2*M12*s1rx*s1ry*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 2*M12*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)))*(-M00*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M01*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M02*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N01*a2 + N02*b2 + p2xc - s1xc)
        res_ref[2,1] = (-M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) - M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))*(M10*s1e*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*M10*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - M11*s1e*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) - 2*M11*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))*(M00*s1e*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*M00*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - M01*s1e*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) - 2*M01*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (M00*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*M00*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)) - M01*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) - 2*M01*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)))*(-M10*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M11*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M12*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N11*a2 + N12*b2 + p2yc - s1yc) + (-M10*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*M10*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)) + M11*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*M11*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)))*(-M00*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M01*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M02*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N01*a2 + N02*b2 + p2xc - s1xc)
        res_ref[2,2] =  N01*(M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1))) + N11*(-M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) - M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))
        res_ref[2,3] = N02*(M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1))) + N12*(-M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) - M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))
        res_ref[3,0] = N00*sign_p2s*(-M00*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)) - M01*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) + M02*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 2*M02*s1rx*s1ry*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 2*M02*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1))) + N10*sign_p2s*(-M10*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)) - M11*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) + M12*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 2*M12*s1rx*s1ry*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 2*M12*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1))) + N20*sign_p2s*(-M20*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)) - M21*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) + M22*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 2*M22*s1rx*s1ry*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 2*M22*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1))) + np.sqrt(N00**2*sign_p2s**2 + N10**2*sign_p2s**2 + N20**2*sign_p2s**2)*((M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))*(-2*M00*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)) - 2*M01*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) + 2*M02*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 4*M02*s1rx*s1ry*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 4*M02*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)))/2 + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))*(-2*M10*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)) - 2*M11*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) + 2*M12*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 4*M12*s1rx*s1ry*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 4*M12*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)))/2 + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))*(-2*M20*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)) - 2*M21*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) + 2*M22*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 4*M22*s1rx*s1ry*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 4*M22*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)))/2)/np.sqrt((M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))**2 + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))**2 + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))**2)
        res_ref[3,1] =  N00*sign_p2s*(-M00*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*M00*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)) + M01*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*M01*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1))) + N10*sign_p2s*(-M10*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*M10*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)) + M11*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*M11*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1))) + N20*sign_p2s*(-M20*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*M20*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)) + M21*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*M21*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1))) + np.sqrt(N00**2*sign_p2s**2 + N10**2*sign_p2s**2 + N20**2*sign_p2s**2)*((M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))*(-2*M00*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 4*M00*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)) + 2*M01*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 4*M01*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)))/2 + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))*(-2*M10*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 4*M10*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)) + 2*M11*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 4*M11*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)))/2 + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))*(-2*M20*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 4*M20*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)) + 2*M21*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 4*M21*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)))/2)/np.sqrt((M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))**2 + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))**2 + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)))**2)
        res_ref[3,2] = 0
        res_ref[3,3] = 0

        print("grad F : res = ",res," res_ref = ",res_ref," ecart = ",np.linalg.norm(res-res_ref))

        return res


    ## TEST f_contacts et grad_f_contacts
    u0 = np.array([2*np.pi/3, 2*np.pi/3, 1, 2])
    r = f_contacts(u0,s1,p2,sign_p2s)
    print("r=",r)  #  [-2.54776551  7.03696638  7.32141373  3.41342755]
    dr = grad_f_contacts(u0,s1,p2,sign_p2s)
    print("dr=",dr)

    def verifier_gradient(f,df,x,s1,p2,sign_p2s):
        N = len(x)
        gg = np.zeros((N,N))
        for i in range(N): # x0, x1, ...
            eps = 1e-6
            e = np.zeros(N)
            e[i] = eps
            gg[:,i] = (f(x+e,s1,p2,sign_p2s) - f(x-e,s1,p2,sign_p2s))/(2*eps)
        print('erreur numerique dans le calcul du gradient: %g (doit etre petit)' % np.linalg.norm(df(x,s1,p2,sign_p2s)-gg))
    verifier_gradient(f_contacts,grad_f_contacts,u0,s1,p2,sign_p2s)


    ## Determination de la donne initiale de la methode de Newton
    # 1/ on calcule les positions sur qq points repartis sur la superellipsoide,
    # 2/ on calcule toutes les distances
    # 3/ on cherche la distance minimale => on a la donnee initiale
    num = 10
    aa = np.linspace(-np.pi/2,np.pi/2,num=num)
    bb = np.linspace(-np.pi,np.pi,num=num)
    pts_ext_s1 = s1.surface_pt(aa,bb)
    pts_ext_p2 = p2.surface_pt(aa,bb)
    distances = np.zeros((bb.shape[0],bb.shape[0]))
    for i in range(bb.shape[0]):
        for j in range(bb.shape[0]):
            distances[i,j] = np.linalg.norm(pts_ext_s1[i,:]-pts_ext_p2[j,:])
    # print("distances = ",distances)
    indmin = np.where(distances==distances.min())
    # print("indmin = ",indmin," min = ",distances.min())
    u0 = np.array( [ aa[indmin[0][0]], bb[indmin[0][0]], aa[indmin[1][0]], bb[indmin[1][0]] ])
    print("u0 = ",u0)

    # ## Methode de Newton de scipy
    # root = fsolve(f_contacts, u0, fprime=grad_f_contacts, xtol=1.0e-14, args=(s1,s2), full_output=True)
    # # root = fsolve(f_contacts, u0, xtol=1.0e-14, args=(s1,s2), full_output=True)
    # b_final = root[0]
    # print("root=",root)  # 1.13408101,  2.18327249, -0.60285452, -2.63898578

    ## Methode de Newton amortie
    # On prend des pas de descente parfois différent de 1 (pour avoir une méthode plus robuste)
    def pas_armijo(u,d,f,gradf,s1,p2,sign_p2s):
        t = 1
        m = np.dot(d,gradf(u,s1,p2,sign_p2s))
        alpha=0.3 # les paramètres alpha et beta comportent une valeur par défaut
        beta=0.5
        while np.linalg.norm(f(u+t*d,s1,p2,sign_p2s)) > np.linalg.norm(f(u,s1,p2,sign_p2s) + alpha*t*m):
            t = beta*t
        return t
    def backtrack(f,x,d,m,s1,p2,sign_p2s,alpha=0.3,beta=0.5):
        t = 1
        while np.linalg.norm(f(x+t*d,s1,p2,sign_p2s)) > np.linalg.norm(f(x,s1,p2,sign_p2s) + alpha*t*m):
            t = beta*t
        return t
    def linesearch(f,x,d,s1,p2,sign_p2s):
        t = 1
        while (np.linalg.norm(f(x+t*d,s1,p2,sign_p2s)) > np.linalg.norm(f(x,s1,p2,sign_p2s)))and (t>0.1) :
            t -= 0.01
        return t
    cc = 0
    itermax = 2000
    u = u0.copy()
    dk = np.ones(u.shape)

    ### test pour comparaison avec le C++
    print("s1 : xc=",s1.xc," yc=",s1.yc," zc=",s1.zc," rx=",s1.rx," ry=",s1.ry," rz=",s1.rz," e=",s1.e," n=",s1.n," q=",s1.q)
    print("p2 : xc=",p2.xc," yc=",p2.yc," zc=",p2.zc," q=",p2.q)
    print("u0 = ",u0," sign_p2s=",sign_p2s)
    val1 = np.array([0.43])
    val2 = np.array([0.65])
    # print("s1.surface_pt(val1,val2)      = ",s1.surface_pt(val1,val2),     " s2.surface_pt(val2,val1)      = ",s2.surface_pt(val2,val1))
    # print("s1.surface_normal(val1,val2)  = ",s1.surface_normal(val1,val2), " s2.surface_normal(val2,val1)  = ",s2.surface_normal(val2,val1))
    # print("s1.surface_tangent(val1,val2) = ",s1.surface_tangent(val1,val2)," s2.surface_tangent(val2,val1) = ",s2.surface_tangent(val2,val1))
    # print("u=",u)
    # print("grad_f_contacts(u,s1,s2) = ",grad_f_contacts(u,s1,s2), " f_contacts(u,s1,s2) = ",f_contacts(u,s1,s2))

    while (cc<itermax) and (np.linalg.norm(dk)>1.0e-7) and (np.linalg.norm(f_contacts(u,s1,p2,sign_p2s))>1e-10) :
        ## dk = -(gradFk)^-1 Fk : direction de descente
        dk = np.linalg.solve(grad_f_contacts(u,s1,p2,sign_p2s), -f_contacts(u,s1,p2,sign_p2s))
        # print("grad Fk = ",grad_f_contacts(u,s1,s2)," -Fk = ",-f_contacts(u,s1,s2))
        ## tk : pas d'armijo
        # tk = pas_armijo(u,dk,f_contacts,grad_f_contacts,s1,s2)
        # tk = backtrack(f_contacts,u,dk,-np.linalg.norm(dk)**2,s1,s2)
        tk = linesearch(f_contacts,u,dk,s1,p2,sign_p2s)
        u += tk*dk
        print("iteration ",cc," dk = ",dk," tk = ",tk," u = ",u," |dk| = ",np.linalg.norm(dk)," np.cost=",np.linalg.norm(f_contacts(u,s1,p2,sign_p2s)))
        cc += 1
    b_final = u

    final_pt1 = s1.surface_pt(np.array([b_final[0]]),np.array([b_final[1]]))
    normal_final_pt1 = s1.surface_normal(np.array([b_final[0]]),np.array([b_final[1]]))
    final_pt2 = p2.surface_pt(np.array([b_final[2]]),np.array([b_final[3]]))
    normal_final_pt2 = p2.surface_normal(np.array([b_final[2]]),np.array([b_final[3]]),sign_p2s)

    print("final_pt1 = ",final_pt1)
    print("final_pt2 = ",final_pt2)

    node_final_pt1 = pv.PolyData(final_pt1)
    node_final_pt1["normal"] = normal_final_pt1
    glyphs_final_pt1 = node_final_pt1.glyph(factor=0.05, geom=pv.Sphere())
    p.add_mesh(glyphs_final_pt1,color="yellow")
    glyphs_normal_final_pt1 = node_final_pt1.glyph(orient="normal",factor=1, geom=pv.Arrow())
    p.add_mesh(glyphs_normal_final_pt1,color="yellow")


    node_final_pt2 = pv.PolyData(final_pt2)
    node_final_pt2["normal"] = normal_final_pt2
    glyphs_final_pt2 = node_final_pt2.glyph(factor=0.05, geom=pv.Sphere())
    p.add_mesh(glyphs_final_pt2,color="yellow")
    glyphs_normal_final_pt2 = node_final_pt2.glyph(orient="normal",factor=1, geom=pv.Arrow())
    p.add_mesh(glyphs_normal_final_pt2,color="yellow")

    line_points = np.row_stack((final_pt1, final_pt2))
    line = pv.PolyData(line_points)
    cells = np.full((len(line_points)-1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(line_points)-1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(line_points), dtype=np.int_)
    line.lines = cells
    line["scalars"] = np.arange(line.n_points)
    tube = line.tube(radius=0.1)
    p.add_mesh(tube, opacity=0.5)

    p.show_grid()
    p.show_bounds()
    p.camera_position = 'xy'
    p.camera.SetParallelProjection(True)
    p.show(window_size=(1600,900))

    # """
    # add_mesh(mesh, color=None, style=None, scalars=None, clim=None, show_edges=None,
    # edge_color=None, point_size=5.0, line_width=None, opacity=1.0, flip_scalars=False,
    # lighting=None, n_colors=256, interpolate_before_map=True, cmap=None, label=None,
    # reset_camera=None, scalar_bar_args=None, show_scalar_bar=None, stitle=None,
    # multi_colors=False, name=None, texture=None, render_points_as_spheres=None,
    # render_lines_as_tubes=False, smooth_shading=None, ambient=0.0, diffuse=1.0,
    # specular=0.0, specular_power=100.0, nan_color=None, nan_opacity=1.0, culling=None,
    # rgb=False, categories=False, use_transparency=False, below_color=None,
    # above_color=None, annotations=None, pickable=True, preference='point',
    # log_scale=False, render=True, **kwargs)
    #
    # pyvista.plot(var_item, off_screen=None, full_screen=False, screenshot=None,
    # interactive=True, cpos=None, window_size=None, show_bounds=False, show_axes=True,
    # notebook=None, background=None, text='', return_img=False, eye_dome_lighting=False,
    # volume=False, parallel_projection=False, use_ipyvtk=None, **kwargs)
    # """
