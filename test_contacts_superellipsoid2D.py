import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyvista as pv
import scipy
import time
from scipy.optimize import fsolve

class SuperEllipsoid2D:

    def __init__(self, x,y, rx,ry, e=1, theta=0):
        """
        En 2D :   |x|**(2/e) + |y|**(2/e) = 1
        Rotation d'angle theta autour du vecteur w : on utilise un quaternion
        En 2D toutes rotations sont autour du vecteur 0,0,1
        0 < e <= 1
        """
        # position
        self.xc = x
        self.yc = y
        self.zc = 0
        # radius
        self.rx = rx
        self.ry = ry
        self.rz = 0
        # shape parameters
        self.e = e  # The “squareness” parameter in the x-y plane
        # quaternion
        self.q = np.array([
            np.cos(theta/2),
            0,
            0,
            np.sin(theta/2)])
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

    def surface_pt(self,b):
        """
        -pi < b < pi
        """
        q0,q1,q2,q3 = self.q
        x = np.sign(np.cos(b))*np.abs(np.cos(b))**self.e
        y = np.sign(np.sin(b))*np.abs(np.sin(b))**self.e
        z = np.zeros((b.shape[0]))
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

    def surface_normal(self,b):
        """
        -pi < b < pi
        """
        q0,q1,q2,q3 = self.q
        nx = np.array( self.ry*np.sign(np.cos(b))*np.abs(np.cos(b))**(2-self.e))
        ny = np.array( self.rx*np.sign(np.sin(b))*np.abs(np.sin(b))**(2-self.e))
        nz = np.zeros((b.shape[0]))
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

    def surface_tangent(self,b):
        """
        -pi < b < pi
        """
        q0,q1,q2,q3 = self.q
        tgtx = np.array( -self.rx*np.sign(np.sin(b))*np.abs(np.sin(b))**(2-self.e))
        tgty = np.array( self.ry*np.sign(np.cos(b))*np.abs(np.cos(b))**(2-self.e))
        tgtz = np.zeros((b.shape[0]))
        ## On applique la rotation
        tangents = np.array([
            tgtx*(q0**2 + q1**2 - q2**2 - q3**2) +
            tgty*(-2*q0*q3 + 2*q1*q2) +
            tgtz*(2*q0*q2 + 2*q1*q3),
            tgtx*(2*q0*q3 + 2*q1*q2) +
            tgty*(q0**2 - q1**2 + q2**2 - q3**2) +
            tgtz*(-2*q0*q1 + 2*q2*q3),
            tgtx*(-2*q0*q2 + 2*q1*q3) +
            tgty*(2*q0*q1 + 2*q2*q3) +
            tgtz*(q0**2 - q1**2 - q2**2 + q3**2)
            ]).T
        nn =  np.linalg.norm(tangents,axis=1)
        tangents[:,0] = tangents[:,0]/nn
        tangents[:,1] = tangents[:,1]/nn
        tangents[:,2] = tangents[:,2]/nn
        # print("\ntangent : repere du solide = \n",np.array([tgtx,tgty,tgtz]).T)
        # print("\ntangent : repere global = \n",tangents)
        return tangents

    def mesh(self):
        mesh = pv.ParametricSuperEllipsoid(
            xradius=1,
            yradius=1,
            zradius=1,
            n1=1,
            n2=self.e
            )
        mesh.transform(self.M())
        return mesh

    def calcul_formel(self):
        import sympy
        from sympy.functions import sign, cos, sin, sqrt
        # quaternion
        q0,q1,q2,q3 = sympy.symbols("q0,q1,q2,q3",real=True)
        # Matrice de rotation définie à partir du quaternion
        Mr = sympy.Matrix([[ q0**2+q1**2-q2**2-q3**2, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2, 0 ],
                           [ 2*q1*q2+2*q0*q3, q0**2-q1**2+q2**2-q3**2, 2*q2*q3-2*q0*q1, 0 ],
                           [ 2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, q0**2-q1**2-q2**2+q3**2, 0 ],
                           [ 0,0,0,1]])
        print("\ncalcul formel : Mr = \n",Mr)
        rx,ry,rz,xc,yc,zc = sympy.symbols("rx,ry,rz,xc,yc,zc",real=True)
        Ms = sympy.Matrix([[rx,0,0,0],[0,ry,0,0],[0,0,rz,0],[0,0,0,1]])
        print("\ncalcul formel : Ms = \n",Ms)
        Mt = sympy.Matrix([[1,0,0,xc],[0,1,0,yc],[0,0,1,zc],[0,0,0,1]])
        print("\ncalcul formel : Mt = \n",Mt)
        M = Mt@Mr@Ms
        print("\ncalcul formel : M = Mt@Mr@Ms = \n",M)
        # Pour un point sur la surface du solide source (i.e. dans le repere de référence
        # avant application des scaling/rotation/translation) on determine les
        # coordonnees du point dans le repere global
        x,y,z = sympy.symbols("x,y,z",real=True)
        pt = M@sympy.Matrix([x,y,z,1])
        print("\ncalcul formel : pt repere global ([x,y,z] dans le repere du solide source) = \n",pt)
        nx,ny,nz = sympy.symbols("nx,ny,nz",real=True)
        ## Vecteur tangent au solide dans le repere global
        b = sympy.symbols("b",real=True)
        # sympy.ask(sympy.Q.is_true(b>-sympy.pi/2),sympy.Q.is_true(b<sympy.pi/2))
        e = sympy.symbols("e",real=True)
        xt = rx*sign(cos(b))*sympy.Abs(cos(b))**e
        yt = ry*sign(sin(b))*sympy.Abs(sin(b))**e
        zt = 0
        dxtdb = sympy.simplify(sympy.diff(xt,b))
        dytdb = sympy.simplify(sympy.diff(yt,b))
        dztdb = sympy.diff(zt,b)
        print("\ncalcul formel : vecteur tangent d (x(b),y(b),z(b)) / db = \n",dxtdb,",\n",dytdb,",\n",dztdb)
        ## Normale exterieure au solide dans le repere global
        normal = Mr@sympy.Matrix([nx,ny,nz,1])
        print("\ncalcul formel : vecteur normal repere global ([nx,ny,nz] dans le repere du solide source) = \n",normal)
        ## Pour la methode de newton
        b1,b2 = sympy.symbols("b1,b2",real=True)
        # superellipsoid1
        s1e = sympy.symbols("s1e",real=True)
        s1q0,s1q1,s1q2,s1q3 = sympy.symbols("s1q0,s1q1,s1q2,s1q3",real=True)
        s1Mr = sympy.Matrix([[ s1q0**2+s1q1**2-s1q2**2-s1q3**2, 2*s1q1*s1q2-2*s1q0*s1q3, 2*s1q1*s1q3+2*s1q0*s1q2, 0 ],
                           [ 2*s1q1*s1q2+2*s1q0*s1q3, s1q0**2-s1q1**2+s1q2**2-s1q3**2, 2*s1q2*s1q3-2*s1q0*s1q1, 0 ],
                           [ 2*s1q1*s1q3-2*s1q0*s1q2, 2*s1q2*s1q3+2*s1q0*s1q1, s1q0**2-s1q1**2-s1q2**2+s1q3**2, 0 ],
                           [ 0,0,0,1]])
        s1rx,s1ry,s1rz,s1xc,s1yc,s1zc = sympy.symbols("s1rx,s1ry,s1rz,s1xc,s1yc,s1zc",real=True)
        s1Ms= sympy.Matrix([[s1rx,0,0,0],[0,s1ry,0,0],[0,0,s1rz,0],[0,0,0,1]])
        s1Mt = sympy.Matrix([[1,0,0,s1xc],[0,1,0,s1yc],[0,0,1,s1zc],[0,0,0,1]])
        s1M = s1Mt@s1Mr@s1Ms
        s1x = sign(cos(b1))*sympy.Abs(cos(b1))**s1e
        s1y = sign(sin(b1))*sympy.Abs(sin(b1))**s1e
        s1z = 0
        pt1 = s1M@sympy.Matrix([s1x,s1y,s1z,1])
        s1nx = s1ry*sign(cos(b1))*sympy.Abs(cos(b1))**(2-s1e)
        s1ny = s1rx*sign(sin(b1))*sympy.Abs(sin(b1))**(2-s1e)
        s1nz = 0
        n1 = s1Mr@sympy.Matrix([s1nx,s1ny,s1nz,1])
        print("\ncalcul formel : pt1 = \n",pt1)
        print("\ncalcul formel : n1 = \n",n1)
        # superellipsoid2
        s2e = sympy.symbols("s2e",real=True)
        s2q0,s2q1,s2q2,s2q3 = sympy.symbols("s2q0,s2q1,s2q2,s2q3",real=True)
        s2Mr = sympy.Matrix([[ s2q0**2+s2q1**2-s2q2**2-s2q3**2, 2*s2q1*s2q2-2*s2q0*s2q3, 2*s2q1*s2q3+2*s2q0*s2q2, 0 ],
                           [ 2*s2q1*s2q2+2*s2q0*s2q3, s2q0**2-s2q1**2+s2q2**2-s2q3**2, 2*s2q2*s2q3-2*s2q0*s2q1, 0 ],
                           [ 2*s2q1*s2q3-2*s2q0*s2q2, 2*s2q2*s2q3+2*s2q0*s2q1, s2q0**2-s2q1**2-s2q2**2+s2q3**2, 0 ],
                           [ 0,0,0,1]])
        s2rx,s2ry,s2rz,s2xc,s2yc,s2zc = sympy.symbols("s2rx,s2ry,s2rz,s2xc,s2yc,s2zc",real=True)
        s2Ms= sympy.Matrix([[s2rx,0,0,0],[0,s2ry,0,0],[0,0,s2rz,0],[0,0,0,1]])
        s2Mt = sympy.Matrix([[1,0,0,s2xc],[0,1,0,s2yc],[0,0,1,s2zc],[0,0,0,1]])
        s2M = s2Mt@s2Mr@s2Ms
        s2x = sign(cos(b2))*sympy.Abs(cos(b2))**s2e
        s2y = sign(sin(b2))*sympy.Abs(sin(b2))**s2e
        s2z = 0
        pt2 = s2M@sympy.Matrix([s2x,s2y,s2z,1])
        s2nx = s2ry*sign(cos(b2))*sympy.Abs(cos(b2))**(2-s2e)
        s2ny = s2rx*sign(sin(b2))*sympy.Abs(sin(b2))**(2-s2e)
        s2nz = 0
        n2 = s2Mr@sympy.Matrix([s2nx,s2ny,s2nz,1])
        print("\ncalcul formel : pt2 = \n",pt2)
        print("\ncalcul formel : n2 = \n",n2)
        # (x2-x1)*ny1 - (y2-y1)*nx1 = 0
        # fct0 = sympy.simplify( (pt2[0]-pt1[0])*n1[1] - (pt2[1]-pt1[1])*n1[0] )
        fct0 =  (pt2[0]-pt1[0])*n1[1] - (pt2[1]-pt1[1])*n1[0]
        print("\ncalcul formel : fct0 = \n",fct0)
        # n1.n2 + ||n1||*||n2|| = 0 (i.e. n1=-n2)
        # fct1 = sympy.simplify( n1[0]*n2[0]+n1[1]*n2[1] )# en considerant n normalisé... +sqrt(n1[0]*n1[0]+n1[1]*n1[1])+sqrt(n2[0]*n2[0]+n2[1]*n2[1])
        fct1 = n1[0]*n2[0]+n1[1]*n2[1] + sqrt(n1[0]*n1[0]+n1[1]*n1[1])*sqrt(n2[0]*n2[0]+n2[1]*n2[1])
        print("\ncalcul formel : fct1 = \n",fct1)
        dfct0db1 = sympy.diff(fct0 ,b1)
        dfct0db2 = sympy.diff(fct0 ,b2)
        dfct1db1 = sympy.diff(fct1 ,b1)
        dfct1db2 = sympy.diff(fct1 ,b2)
        print("\ncalcul formel : dfct0db1 = \n",dfct0db1)
        print("\ncalcul formel : dfct0db2 = \n",dfct0db2)
        print("\ncalcul formel : dfct1db1 = \n",dfct1db1)
        print("\ncalcul formel : dfct1db2 = \n",dfct1db2)

if __name__ == '__main__':

    # Calcul de la distance entre deux superellipsoides
    # (1) appartenance à la superellipsoide 1 => b1 => x1=x(b1), y1=y(b1)
    # (2) appartenance à la superellipsoide 2 => b2 => x2=x(b2), y2=y(b2)
    # (3) soient X1=(x1,y1) X2=(x2,y2)
    #     on veut X2X1 ^ n1 = 0 (i.e. X2X1 et n1 colineaires)
    #     (x2-x1)   nx1
    #     (y2-y1)   ny1    i.e. (x2-x1)*ny1 - (y2-y1)*nx1 = 0
    #        0      0
    #     (x2-x1)   nx1
    #     (y2-y1)   ny1
    # (4) normales opposées : n1.n2 + ||n1||*||n2|| = 0 (i.e. n1=-n2)
    def f_contacts(u,s1,s2):
        b1,b2 = u
        res = np.zeros((2,))
        s1q0,s1q1,s1q2,s1q3 = s1.q
        s1e = s1.e ; s1rx = s1.rx ; s1ry = s1.ry ; s1rz = s1.rz ; s1xc = s1.xc ; s1yc = s1.yc ; s1zc = s1.zc
        s2q0,s2q1,s2q2,s2q3 = s2.q
        s2e = s2.e ; s2rx = s2.rx ; s2ry = s2.ry ; s2rz = s2.rz ; s2xc = s2.xc ; s2yc = s2.yc ; s2zc = s2.zc
        # ## res_ref est directement obtenu a partir du calcul formel
        # ## permet de tester que la reecriture des formules n'a pas fait ajouter des erreurs
        # res_ref = np.zeros((2,))
        # res_ref[0] =  -(s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*\
        # (-s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**s1e*np.sign(np.cos(b1)) - s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**s1e*np.sign(np.sin(b1)) - s1yc + s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(b2))**s2e*np.sign(np.cos(b2)) + s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**s2e*np.sign(np.sin(b2)) + s2yc) + \
        # (s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*\
        # (-s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**s1e*np.sign(np.cos(b1)) - s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**s1e*np.sign(np.sin(b1)) - s1xc + s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(b2))**s2e*np.sign(np.cos(b2)) + s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**s2e*np.sign(np.sin(b2)) + s2xc)
        # res_ref[1] = (s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*\
        # (s2rx*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))) + \
        # (s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*\
        # (s2rx*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))) + \
        # np.sqrt((s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2)*\
        # np.sqrt((s2rx*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2)


        # [ q0**2+q1**2-q2**2-q3**2, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2, 0 ],
        # [ 2*q1*q2+2*q0*q3, q0**2-q1**2+q2**2-q3**2, 2*q2*q3-2*q0*q1, 0 ],
        # [ 2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, q0**2-q1**2-q2**2+q3**2 ]

        s1q00 = s1q0**2
        s1q11 = s1q1**2
        s1q22 = s1q2**2
        s1q33 = s1q3**2
        s1q03 = 2*s1q0*s1q3
        s1q12 = 2*s1q1*s1q2
        s2q00 = s2q0**2
        s2q11 = s2q1**2
        s2q22 = s2q2**2
        s2q33 = s2q3**2
        s2q03 = 2*s2q0*s2q3
        s2q12 = 2*s2q1*s2q2
        Q1 = (s1q00 + s1q11 - s1q22 - s1q33)  # M1[0,0]
        R1 = (s2q00 + s2q11 - s2q22 - s2q33)  # M2[0,0]
        Q2 = (s1q00 - s1q11 + s1q22 - s1q33)  # M1[1,1]
        R2 = (s2q00 - s2q11 + s2q22 - s2q33)  # M2[1,1]
        Q3 = (s1q03 + s1q12)  # M1[1,0]
        R3 = (s2q03 + s2q12)  # M2[1,0]
        Q4 = (-s1q03 + s1q12) # M1[0,1]
        R4 = (-s2q03 + s2q12) # M2[0,1]
        sb2e1 = np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1))
        sbe1 = np.abs(np.sin(b1))**s1e*np.sign(np.sin(b1))
        cb2e1 = np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))
        cbe1 = np.abs(np.cos(b1))**s1e*np.sign(np.cos(b1))
        sb2e2 = np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2))
        sbe2 = np.abs(np.sin(b2))**s2e*np.sign(np.sin(b2))
        cb2e2 = np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))
        cbe2 = np.abs(np.cos(b2))**s2e*np.sign(np.cos(b2))
        res[0] =  -( s1rx*Q4*sb2e1 + s1ry*Q1*cb2e1 ) * \
        ( -s1rx*Q3*cbe1 - s1ry*Q2*sbe1 - s1yc + s2rx*R3*cbe2 + s2ry*R2*sbe2 + s2yc ) + \
        ( s1rx*Q2*sb2e1 + s1ry*Q3*cb2e1) * \
        ( -s1rx*Q1*cbe1 - s1ry*Q4*sbe1 - s1xc + s2rx*R1*cbe2 + s2ry*R4*sbe2 + s2xc )
        res[1] = ( s1rx*Q4*sb2e1 + s1ry*Q1*cb2e1 ) * \
        ( s2rx*R4*sb2e2 + s2ry*R1*cb2e2 ) + \
        ( s1rx*Q2*sb2e1 + s1ry*Q3*cb2e1 ) * \
        ( s2rx*R2*sb2e2 + s2ry*R3*cb2e2 ) + \
        np.sqrt( (s1rx*Q4*sb2e1 + s1ry*Q1*cb2e1)**2 + (s1rx*Q2*sb2e1 + s1ry*Q3*cb2e1)**2 ) * \
        np.sqrt( (s2rx*R4*sb2e2 + s2ry*R1*cb2e2)**2 + (s2rx*R2*sb2e2 + s2ry*R3*cb2e2)**2 )
        # print("F : res = ",res," res_ref = ",res_ref," ecart = ",np.linalg.norm(res-res_ref))
        return res

    def DiracDelta(x):
        return 0

    def grad_f_contacts(u,s1,s2):
        b1,b2 = u
        s1q0,s1q1,s1q2,s1q3 = s1.q
        s1e = s1.e ; s1rx = s1.rx ; s1ry = s1.ry ; s1rz = s1.rz ; s1xc = s1.xc ; s1yc = s1.yc ; s1zc = s1.zc
        s2q0,s2q1,s2q2,s2q3 = s2.q
        s2e = s2.e ; s2rx = s2.rx ; s2ry = s2.ry ; s2rz = s2.rz ; s2xc = s2.xc ; s2yc = s2.yc ; s2zc = s2.zc
        res = np.zeros((2,2))
        s1q00 = s1q0**2
        s1q11 = s1q1**2
        s1q22 = s1q2**2
        s1q33 = s1q3**2
        s1q03 = 2*s1q0*s1q3
        s1q12 = 2*s1q1*s1q2
        s2q00 = s2q0**2
        s2q11 = s2q1**2
        s2q22 = s2q2**2
        s2q33 = s2q3**2
        s2q03 = 2*s2q0*s2q3
        s2q12 = 2*s2q1*s2q2
        Q1 = (s1q00 + s1q11 - s1q22 - s1q33)
        R2 = (s2q00 - s2q11 + s2q22 - s2q33)
        R1 = (s2q00 + s2q11 - s2q22 - s2q33)
        Q2 = (s1q00 - s1q11 + s1q22 - s1q33)
        Q4 = (-s1q03 + s1q12)
        R3 = (s2q03 + s2q12)
        Q3 = (s1q03 + s1q12)
        R4 = (-s2q03 + s2q12)
        sb2e1 = np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1))
        sbe1 = np.abs(np.sin(b1))**s1e*np.sign(np.sin(b1))
        cb2e1 = np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))
        cbe1 = np.abs(np.cos(b1))**s1e*np.sign(np.cos(b1))
        sb2e2 = np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2))
        sbe2 = np.abs(np.sin(b2))**s2e*np.sign(np.sin(b2))
        cb2e2 = np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))
        cbe2 = np.abs(np.cos(b2))**s2e*np.sign(np.cos(b2))
        scb1e1 = np.sin(b1)*np.abs(np.cos(b1))**(s1e - 1)
        csb1e1 = np.cos(b1)*np.abs(np.sin(b1))**(s1e - 1)
        iscb1e1 = np.sin(b1)*np.abs(np.cos(b1))**(1 - s1e)
        icsb1e1 = np.cos(b1)*np.abs(np.sin(b1))**(1 - s1e)
        scb1e2 = np.sin(b2)*np.abs(np.cos(b2))**(s2e - 1)
        csb1e2 = np.cos(b2)*np.abs(np.sin(b2))**(s2e - 1)
        iscb1e2 = np.sin(b2)*np.abs(np.cos(b2))**(1 - s2e)
        icsb1e2 = np.cos(b2)*np.abs(np.sin(b2))**(1 - s2e)
        res[0,0] =  ( -s1rx*Q4*sb2e1 - s1ry*Q1*cb2e1 ) * \
        ( s1e*s1rx*Q3*scb1e1 - s1e*s1ry*Q2*csb1e1 ) + \
        ( s1rx*Q2*sb2e1 + s1ry*Q3*cb2e1) * \
        ( s1e*s1rx*Q1*scb1e1 - s1e*s1ry*Q4*csb1e1 ) + \
        ( -s1rx*(2 - s1e)*Q4*icsb1e1 + s1ry*(2 - s1e)*Q1*iscb1e1 ) * \
        ( -s1rx*Q3*cbe1 - s1ry*Q2*sbe1 - s1yc + s2rx*R3*cbe2 + s2ry*R2*sbe2 + s2yc) + \
        ( s1rx*(2 - s1e)*Q2*icsb1e1 - s1ry*(2 - s1e)*Q3*iscb1e1 ) * \
        ( -s1rx*Q1*cbe1 - s1ry*Q4*sbe1 - s1xc + s2rx*R1*cbe2 + s2ry*R4*sbe2 + s2xc )
        res[0,1] = ( -s1rx*Q4*sb2e1 - s1ry*Q1*cb2e1 ) * \
        ( -s2e*s2rx*R3*scb1e2 + s2e*s2ry*R2*csb1e2 ) + \
        ( s1rx*Q2*sb2e1 + s1ry*Q3*cb2e1 ) * \
        ( -s2e*s2rx*R1*scb1e2 + s2e*s2ry*R4*csb1e2 )
        res[1,0] =  (
        ( s1rx*Q4*sb2e1 + s1ry*Q1*cb2e1 ) * \
        ( 2*s1rx*(2 - s1e)*Q4*icsb1e1 - 2*s1ry*(2 - s1e)*Q1*iscb1e1 )/2 + \
        ( s1rx*Q2*sb2e1 + s1ry*Q3*cb2e1) * \
        ( 2*s1rx*(2 - s1e)*Q2*icsb1e1 - 2*s1ry*(2 - s1e)*Q3*iscb1e1 )/2
        ) * \
        np.sqrt(
        ( s2rx*R4*sb2e2 + s2ry*R1*cb2e2)**2 + \
        ( s2rx*R2*sb2e2 + s2ry*R3*cb2e2)**2
        ) / np.sqrt(
        ( s1rx*Q4*sb2e1 + s1ry*Q1*cb2e1)**2 + \
        ( s1rx*Q2*sb2e1 + s1ry*Q3*cb2e1)**2
        ) + \
        ( s2rx*R4*sb2e2 + s2ry*R1*cb2e2 ) * \
        ( s1rx*(2 - s1e)*Q4*icsb1e1 - s1ry*(2 - s1e)*Q1*iscb1e1 ) + \
        ( s2rx*R2*sb2e2 + s2ry*R3*cb2e2) * \
        ( s1rx*(2 - s1e)*Q2*icsb1e1 - s1ry*(2 - s1e)*Q3*iscb1e1 )
        res[1,1] = (
        (s2rx*R4*sb2e2 + s2ry*R1*cb2e2) * \
        (2*s2rx*(2 - s2e)*R4*icsb1e2 - 2*s2ry*(2 - s2e)*R1*iscb1e2 )/2 + \
        (s2rx*R2*sb2e2 + s2ry*R3*cb2e2) * \
        (2*s2rx*(2 - s2e)*R2*icsb1e2 - 2*s2ry*(2 - s2e)*R3*iscb1e2 )/2) * \
        np.sqrt(
        (s1rx*Q4*sb2e1 + s1ry*Q1*cb2e1)**2 + \
        (s1rx*Q2*sb2e1 + s1ry*Q3*cb2e1)**2
        )/np.sqrt(
        (s2rx*R4*sb2e2 + s2ry*R1*cb2e2)**2 + \
        (s2rx*R2*sb2e2 + s2ry*R3*cb2e2)**2
        ) + \
        ( s1rx*Q4*sb2e1 + s1ry*Q1*cb2e1 ) * \
        ( s2rx*(2 - s2e)*R4*icsb1e2 - s2ry*(2 - s2e)*R1*iscb1e2 ) + \
        ( s1rx*Q2*sb2e1 + s1ry*Q3*cb2e1 ) * \
        ( s2rx*(2 - s2e)*R2*icsb1e2 - s2ry*(2 - s2e)*R3*iscb1e2 )
        # ## res_ref est directement obtenu a partir du calcul formel
        # ## permet de tester que la reecriture des formules n'a pas fait ajouter des erreurs
        # res_ref = np.zeros((2,2))
        # res_ref[0,0] =  (-s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) - s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1e*s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(b1))**s1e*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - s1e*s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1)) - 2*s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*DiracDelta(np.sin(b1))) + (s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1e*s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(b1))**s1e*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - s1e*s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1)) - 2*s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*DiracDelta(np.sin(b1))) + (-s1rx*(2 - s1e)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) - 2*s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*DiracDelta(np.sin(b1)) + s1ry*(2 - s1e)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))*(-s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**s1e*np.sign(np.cos(b1)) - s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**s1e*np.sign(np.sin(b1)) - s1yc + s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(b2))**s2e*np.sign(np.cos(b2)) + s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**s2e*np.sign(np.sin(b2)) + s2yc) + (s1rx*(2 - s1e)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*DiracDelta(np.sin(b1)) - s1ry*(2 - s1e)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))*(-s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**s1e*np.sign(np.cos(b1)) - s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**s1e*np.sign(np.sin(b1)) - s1xc + s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(b2))**s2e*np.sign(np.cos(b2)) + s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**s2e*np.sign(np.sin(b2)) + s2xc)
        # res_ref[0,1] = (-s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) - s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2e*s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(b2))**s2e*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) + s2e*s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) - 2*s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(b2)) + 2*s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*DiracDelta(np.sin(b2))) + (s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2e*s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(b2))**s2e*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) + s2e*s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) - 2*s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(b2)) + 2*s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*DiracDelta(np.sin(b2)))
        # res_ref[1,0] =  ((s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(2*s1rx*(2 - s1e)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 4*s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*DiracDelta(np.sin(b1)) - 2*s1ry*(2 - s1e)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 4*s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))/2 + (s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(2*s1rx*(2 - s1e)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 4*s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*DiracDelta(np.sin(b1)) - 2*s1ry*(2 - s1e)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 4*s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))/2)*np.sqrt((s2rx*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2)/np.sqrt((s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2) + (s2rx*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(s1rx*(2 - s1e)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*DiracDelta(np.sin(b1)) - s1ry*(2 - s1e)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1))) + (s2rx*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(s1rx*(2 - s1e)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*DiracDelta(np.sin(b1)) - s1ry*(2 - s1e)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))
        # res_ref[1,1] = ((s2rx*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(2*s2rx*(2 - s2e)*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) + 4*s2rx*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*DiracDelta(np.sin(b2)) - 2*s2ry*(2 - s2e)*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 4*s2ry*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2)))/2 + (s2rx*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(2*s2rx*(2 - s2e)*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) + 4*s2rx*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*DiracDelta(np.sin(b2)) - 2*s2ry*(2 - s2e)*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 4*s2ry*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2)))/2)*np.sqrt((s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2)/np.sqrt((s2rx*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2) + (s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s2rx*(2 - s2e)*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) + 2*s2rx*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*DiracDelta(np.sin(b2)) - s2ry*(2 - s2e)*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 2*s2ry*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2))) + (s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s2rx*(2 - s2e)*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) + 2*s2rx*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*DiracDelta(np.sin(b2)) - s2ry*(2 - s2e)*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 2*s2ry*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2)))
        # print("grad F : res = ",res," res_ref = ",res_ref," ecart = ",np.linalg.norm(res-res_ref))
        return res

    def linesearch(f,x,d,s1,s2):
        t = 1
        while (np.linalg.norm(f(x+t*d,s1,s2)) > np.linalg.norm(f(x,s1,s2)))and (t>0.1) :
            t -= 0.01
        return t

    def verifier_gradient(f,df,x,s1,s2):
        N = len(x)
        gg = np.zeros((N,N))
        for i in range(N): # x0, x1, ...
            eps = 1e-6
            e = np.zeros(N)
            e[i] = eps
            gg[:,i] = (f(x+e,s1,s2) - f(x-e,s1,s2))/(2*eps)
        print('erreur numerique dans le calcul du gradient: %g (doit etre petit)' % np.linalg.norm(df(x,s1,s2)-gg))

    #verifier_gradient(f_contacts,grad_f_contacts,np.array([2*np.pi/3, np.pi/3]),s1,s2)

    N = 180
    tgrid = np.linspace(0,1,N)
    thetagrid = np.linspace(0,180,N)
    R = 10
    s1 = SuperEllipsoid2D(1,-2, 1,1, e=0.5, theta=80)
    #s1.calcul_formel()

    # p = pv.Plotter(window_size=[2400,1350],off_screen=True)
    p = pv.Plotter(window_size=[2400,1350])

    L = 20
    grid = pv.UniformGrid()
    arr = np.arange((2*L)**2).reshape((2*L,2*L,1))
    grid.dimensions = np.array(arr.shape) + 1 #dim + 1 because cells
    grid.origin = (-L, -L, 0)
    grid.spacing = (1, 1, 0)
    p.add_mesh(grid, show_edges=True, opacity=0.1)

    bb = np.linspace(-np.pi,np.pi,num=20)
    # bb = np.array([-np.pi, -np.pi/2, 0.001, np.pi/2])

    p.add_mesh(s1.mesh(),color="red", opacity=0.5, name="s1")

    pts1 = s1.surface_pt(bb)
    normals1 = s1.surface_normal(bb)
    tangents1 = s1.surface_tangent(bb)
    nodes1 = pv.PolyData(pts1)
    nodes1["normal"] = normals1
    nodes1["tangent"] = tangents1
    p.add_mesh(nodes1.glyph(factor=0.05, geom=pv.Sphere()),color="blue", name="pts1")
    p.add_mesh(nodes1.glyph(orient="normal",factor=0.25, geom=pv.Arrow()),color="blue", name="normals1")
    p.add_mesh(nodes1.glyph(orient="tangent",factor=0.25, geom=pv.Arrow()),color="magenta", name="tgts1")

    for it,t in enumerate(tgrid):
        xc = 2 + R*np.cos(2*np.pi*t)
        yc = 4 + R*np.sin(2*np.pi*t)
        theta = thetagrid[it]
        s2 = SuperEllipsoid2D(xc,yc, 2,3, e=1, theta=theta)

        ## Determination de la donne initiale de la methode de Newton
        # 1/ on calcule les positions des 4 "extremites" des deux superellipsoid2D,
        # 2/ on calcule toutes les distances
        # 3/ on cherche la distance minimale => on a la donnee initiale
        # binit = np.linspace(-np.pi,np.pi,num=20)
        binit = np.array([-np.pi, -np.pi/2, 0.001, np.pi/2])
        pts_ext_s1 = s1.surface_pt(binit)
        pts_ext_s2 = s2.surface_pt(binit)
        distances = np.zeros((binit.shape[0],binit.shape[0]))
        for i in range(binit.shape[0]):
            for j in range(binit.shape[0]):
                distances[i,j] = np.linalg.norm(pts_ext_s1[i,:]-pts_ext_s2[j,:])
        # print("distances = ",distances)
        indmin = np.where(distances==distances.min())
        u0 = np.array( [ binit[indmin[0][0]], binit[indmin[1][0]] ])
        # print("u0 = ",u0)

        ## Methode de Newton de scipy
        # root = fsolve(f_contacts, u0, fprime=grad_f_contacts, xtol=1.0e-14, args=(s1,s2), full_output=True)
        # b_final = root[0]
        # print("root=",root)

        ## Methode de Newton amortie
        # On prend des pas de descente parfois différent de 1 (pour avoir une méthode plus robuste)
        cc = 0
        itermax = 200
        u = u0.copy()
        dk = np.ones(u.shape)
        ### test pour comparaison avec le C++
        # print("s1 : xc=",s1.xc," yc=",s1.yc," rx=",s1.rx," ry=",s1.ry," e=",s1.e," q=",s1.q)
        # print("s2 : xc=",s2.xc," yc=",s2.yc," rx=",s2.rx," ry=",s2.ry," e=",s2.e," q=",s2.q)
        # print("u0 = ",u0)
        # val1 = np.array([0.43])
        # val2 = np.array([0.65])
        # print("s1.surface_pt(val1)      = ",s1.surface_pt(val1),     " s2.surface_pt(val2)      = ",s2.surface_pt(val2))
        # print("s1.surface_normal(val1)  = ",s1.surface_normal(val1), " s2.surface_normal(val2)  = ",s2.surface_normal(val2))
        # print("s1.surface_tangent(val1) = ",s1.surface_tangent(val1)," s2.surface_tangent(val2) = ",s2.surface_tangent(val2))
        # print("grad_f_contacts(u,s1,s2) = ",grad_f_contacts(u,s1,s2), " f_contacts(u,s1,s2) = ",f_contacts(u,s1,s2))
        # sys.exit()

        while (cc<itermax) and (np.linalg.norm(dk)>1.0e-7) and (np.linalg.norm(f_contacts(u,s1,s2))>1e-10) :
            ## dk = -(gradFk)^-1 Fk : direction de descente
            dk = np.linalg.solve(grad_f_contacts(u,s1,s2), -f_contacts(u,s1,s2))
            # print("grad Fk = ",grad_f_contacts(u,s1,s2)," -Fk = ",-f_contacts(u,s1,s2))
            ## tk : pas d'armijo
            # tk = pas_armijo(u,dk,f_contacts,grad_f_contacts,s1,s2)
            # tk = backtrack(f_contacts,u,dk,-np.linalg.norm(dk)**2,s1,s2)
            tk = linesearch(f_contacts,u,dk,s1,s2)
            u += tk*dk
            # print("iteration ",cc," dk = ",dk," tk = ",tk," u = ",u," |dk| = ",np.linalg.norm(dk)," cost=",np.linalg.norm(f_contacts(u,s1,s2)))
            cc += 1
        print("it = ",it," nb iterations = ",cc," |dk| = ",np.linalg.norm(dk)," cost=",np.linalg.norm(f_contacts(u,s1,s2)))
        p.add_text("nb iterations = "+str(cc),position="upper_left",name="titre")
        b_final = u

        print("u final = ",u)



        p.add_mesh(s2.mesh(),color="green", opacity=0.5, name="s2")

        pts2 = s2.surface_pt(bb)
        normals2 = s2.surface_normal(bb)
        tangents2 = s2.surface_tangent(bb)
        nodes2 = pv.PolyData(pts2)
        nodes2["normal"] = normals2
        nodes2["tangent"] = tangents2
        p.add_mesh(nodes2.glyph(factor=0.05, geom=pv.Sphere()),color="blue", name="pts2")
        p.add_mesh(nodes2.glyph(orient="normal",factor=0.25, geom=pv.Arrow()),color="blue", name="normals2")
        p.add_mesh(nodes2.glyph(orient="tangent",factor=0.25, geom=pv.Arrow()),color="magenta", name="tgts2")

        final_pt1 = s1.surface_pt(np.array( [b_final[0]] ))
        normal_final_pt1 = s1.surface_normal(np.array( [b_final[0]] ))
        final_pt2 = s2.surface_pt(np.array( [b_final[1]] ))
        normal_final_pt2 = s2.surface_normal(np.array( [b_final[1]] ))
        # print("final_pt1 = ",final_pt1)
        # print("final_pt2 = ",final_pt2)
        # sys.exit()
        node_final_pt1 = pv.PolyData(final_pt1)
        node_final_pt1["normal"] = normal_final_pt1
        p.add_mesh(node_final_pt1.glyph(factor=0.05, geom=pv.Sphere()),color="pink",name="fpt1")
        p.add_mesh(node_final_pt1.glyph(orient="normal",factor=1, geom=pv.Arrow()),color="pink",name="fn1")

        node_final_pt2 = pv.PolyData(final_pt2)
        node_final_pt2["normal"] = normal_final_pt2
        p.add_mesh(node_final_pt2.glyph(factor=0.05, geom=pv.Sphere()),color="yellow",name="fpt2")
        p.add_mesh(node_final_pt2.glyph(orient="normal",factor=1, geom=pv.Arrow()),color="yellow",name="fn2")

        # line_points = np.row_stack((final_pt1, final_pt2))
        # line = pv.PolyData(line_points)
        # cells = np.full((len(line_points)-1, 3), 2, dtype=np.int_)
        # cells[:, 1] = np.arange(0, len(line_points)-1, dtype=np.int_)
        # cells[:, 2] = np.arange(1, len(line_points), dtype=np.int_)
        # line.lines = cells
        # line["scalars"] = np.arange(line.n_points)
        # tube = line.tube(radius=0.1)
        # p.add_mesh(tube, opacity=0.5,name="tube",show_scalar_bar=False)
        p.add_mesh(pv.Line(pointa=final_pt1,pointb=final_pt2,resolution=2),color="red",name="line")
        if (it == 0):
            p.show(auto_close=False, cpos="xy")
            # p.show(auto_close=False, cpos="xy",screenshot='image_'+str(it)+'.png')
        else:
            p.render()
            # p.show(auto_close=False, cpos="xy",screenshot='image_'+str(it)+'.png')
            time.sleep(0.5)


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
