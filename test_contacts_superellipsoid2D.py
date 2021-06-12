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
        res_ref = np.zeros((2,))
        res_ref[0] =  -(s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*\
        (-s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**s1e*np.sign(np.cos(b1)) - s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**s1e*np.sign(np.sin(b1)) - s1yc + s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(b2))**s2e*np.sign(np.cos(b2)) + s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**s2e*np.sign(np.sin(b2)) + s2yc) + \
        (s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*\
        (-s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**s1e*np.sign(np.cos(b1)) - s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**s1e*np.sign(np.sin(b1)) - s1xc + s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(b2))**s2e*np.sign(np.cos(b2)) + s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**s2e*np.sign(np.sin(b2)) + s2xc)
        res_ref[1] = (s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*\
        (s2rx*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))) + \
        (s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*\
        (s2rx*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))) + \
        np.sqrt((s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2)*\
        np.sqrt((s2rx*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2)

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
        M00 = (s1q00 + s1q11 - s1q22 - s1q33)
        M01 = (-s1q03 + s1q12)
        M10 = (s1q03 + s1q12)
        M11 = (s1q00 - s1q11 + s1q22 - s1q33)
        N00 = (s2q00 + s2q11 - s2q22 - s2q33)
        N01 = (-s2q03 + s2q12)
        N10 = (s2q03 + s2q12)
        N11 = (s2q00 - s2q11 + s2q22 - s2q33)

        cb1 = np.cos(b1)
        cb1e = np.sign(cb1)*np.abs(cb1)**s1e
        cb1e2 = np.sign(cb1)*np.abs(cb1)**(2 - s1e)

        sb1 = np.sin(b1)
        sb1e = np.sign(sb1)*np.abs(sb1)**s1e
        sb1e2 = np.sign(sb1)*np.abs(sb1)**(2 - s1e)

        F1 = s1rx * sb1e2
        F2 = s1ry * sb1e
        F3 = s1ry * cb1e2
        F4 = s1rx * cb1e

        cb2 = np.cos(b2)
        cb2e = np.sign(cb2)*np.abs(cb2)**s2e
        cb2e2 = np.sign(cb2)*np.abs(cb2)**(2 - s2e)

        sb2 = np.sin(b2)
        sb2e = np.sign(sb2)*np.abs(sb2)**s2e
        sb2e2 = np.sign(sb2)*np.abs(sb2)**(2 - s2e)

        F5 = s2rx * sb2e2
        F6 = s2ry * sb2e
        F7 = s2ry * cb2e2
        F8 = s2rx * cb2e

        res[0] =  -( M01*F1 + M00*F3 ) * ( -M10*F4 - M11*F2 - s1yc + N10*F8 + N11*F6 + s2yc ) + \
                   ( M11*F1 + M10*F3 ) * ( -M00*F4 - M01*F2 - s1xc + N00*F8 + N01*F6 + s2xc )
        res[1] = ( M01*F1 + M00*F3 ) * ( N01*F5 + N00*F7 ) + ( M11*F1 + M10*F3 ) * ( N11*F5 + N10*F7 ) + \
                 np.sqrt( (M01*F1 + M00*F3)**2 + (M11*F1 + M10*F3)**2 ) * np.sqrt( (N01*F5 + N00*F7)**2 + (N11*F5 + N10*F7)**2 )
        print("F : res = ",res," res_ref = ",res_ref," ecart = ",np.linalg.norm(res-res_ref))
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
        M00 = (s1q00 + s1q11 - s1q22 - s1q33)
        M01 = (-s1q03 + s1q12)
        M10 = (s1q03 + s1q12)
        M11 = (s1q00 - s1q11 + s1q22 - s1q33)
        N00 = (s2q00 + s2q11 - s2q22 - s2q33)
        N01 = (-s2q03 + s2q12)
        N10 = (s2q03 + s2q12)
        N11 = (s2q00 - s2q11 + s2q22 - s2q33)

        cb1 = np.cos(b1)
        cb1e = np.sign(cb1)*np.abs(cb1)**s1e
        cb1e2 = np.sign(cb1)*np.abs(cb1)**(2 - s1e)

        sb1 = np.sin(b1)
        sb1e = np.sign(sb1)*np.abs(sb1)**s1e
        sb1e2 = np.sign(sb1)*np.abs(sb1)**(2 - s1e)

        F1 = s1rx * sb1e2
        F2 = s1ry * sb1e
        F3 = s1ry * cb1e2
        F4 = s1rx * cb1e

        cb2 = np.cos(b2)
        cb2e = np.sign(cb2)*np.abs(cb2)**s2e
        cb2e2 = np.sign(cb2)*np.abs(cb2)**(2 - s2e)

        sb2 = np.sin(b2)
        sb2e = np.sign(sb2)*np.abs(sb2)**s2e
        sb2e2 = np.sign(sb2)*np.abs(sb2)**(2 - s2e)

        F5 = s2rx * sb2e2
        F6 = s2ry * sb2e
        F7 = s2ry * cb2e2
        F8 = s2rx * cb2e

        F9 = s1e * s1rx * sb1 * np.abs(cb1)**(s1e - 1)
        F10 = s1e * s1ry * cb1 * np.abs(sb1)**(s1e - 1)
        F11 = s1ry * (2 - s1e) * sb1 * np.abs(cb1)**(1 - s1e)
        F12 = s1rx * (2 - s1e) * cb1 * np.abs(sb1)**(1 - s1e)
        F13 = s2e * s2rx * sb2 * np.abs(cb2)**(s2e - 1)
        F14 = s2e * s2ry * cb2 * np.abs(sb2)**(s2e - 1)
        F15 = s2ry * (2 - s2e) * sb2 * np.abs(cb2)**(1 - s2e)
        F16 = s2rx * (2 - s2e) * cb2 * np.abs(sb2)**(1 - s2e)
        res[0,0] =  ( -M01*F1 - M00*F3 ) * ( M10*F9 - M11*F10 ) + \
                    (  M11*F1 + M10*F3 ) * ( M00*F9 - M01*F10 ) + \
                    ( -M01*F12 + M00*F11 ) * ( -M10*F4 - M11*F2 - s1yc + N10*F8 + N11*F6 + s2yc ) + \
                    (  M11*F12 - M10*F11 ) * ( -M00*F4 - M01*F2 - s1xc + N00*F8 + N01*F6 + s2xc )
        res[0,1] = ( -M01*F1 - M00*F3 ) * ( -N10*F13 + N11*F14 ) + \
                   (  M11*F1 + M10*F3 ) * ( -N00*F13 + N01*F14 )
        res[1,0] =  ( ( M01*F1 + M00*F3 ) * ( M01*F12 - M00*F11 ) + \
                      ( M11*F1 + M10*F3) * ( M11*F12 - M10*F11 ) \
                    ) \
        * np.sqrt( ( N01*F5 + N00*F7)**2 + ( N11*F5 + N10*F7)**2 ) \
        / np.sqrt( ( M01*F1 + M00*F3)**2 + ( M11*F1 + M10*F3)**2 ) \
        + ( N01*F5 + N00*F7 ) * ( M01*F12 - M00*F11 ) \
        + ( N11*F5 + N10*F7 ) * ( M11*F12 - M10*F11 )
        res[1,1] = ( (N01*F5 + N00*F7) * (N01*F16 - N00*F15 ) \
        + (N11*F5 + N10*F7) * (N11*F16 - N10*F15 ) ) \
        * np.sqrt( ( M01*F1 + M00*F3 )**2 + ( M11*F1 + M10*F3 )**2 ) \
        / np.sqrt( ( N01*F5 + N00*F7 )**2 + ( N11*F5 + N10*F7 )**2 ) \
        + ( M01*F1 + M00*F3 ) * ( N01*F16 - N00*F15 ) + ( M11*F1 + M10*F3 ) * ( N11*F16 - N10*F15 )
        # ## res_ref est directement obtenu a partir du calcul formel
        # ## permet de tester que la reecriture des formules n'a pas fait ajouter des erreurs
        res_ref = np.zeros((2,2))
        res_ref[0,0] =  (-s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) - s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1e*s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(b1))**s1e*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - s1e*s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1)) - 2*s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*DiracDelta(np.sin(b1))) + (s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1e*s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(b1))**s1e*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - s1e*s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1)) - 2*s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*DiracDelta(np.sin(b1))) + (-s1rx*(2 - s1e)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) - 2*s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*DiracDelta(np.sin(b1)) + s1ry*(2 - s1e)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))*(-s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**s1e*np.sign(np.cos(b1)) - s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**s1e*np.sign(np.sin(b1)) - s1yc + s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(b2))**s2e*np.sign(np.cos(b2)) + s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**s2e*np.sign(np.sin(b2)) + s2yc) + (s1rx*(2 - s1e)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*DiracDelta(np.sin(b1)) - s1ry*(2 - s1e)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))*(-s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**s1e*np.sign(np.cos(b1)) - s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**s1e*np.sign(np.sin(b1)) - s1xc + s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(b2))**s2e*np.sign(np.cos(b2)) + s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**s2e*np.sign(np.sin(b2)) + s2xc)
        res_ref[0,1] = (-s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) - s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2e*s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(b2))**s2e*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) + s2e*s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) - 2*s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(b2)) + 2*s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*DiracDelta(np.sin(b2))) + (s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2e*s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(b2))**s2e*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) + s2e*s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) - 2*s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(b2)) + 2*s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*DiracDelta(np.sin(b2)))
        res_ref[1,0] =  ((s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(2*s1rx*(2 - s1e)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 4*s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*DiracDelta(np.sin(b1)) - 2*s1ry*(2 - s1e)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 4*s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))/2 + (s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(2*s1rx*(2 - s1e)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 4*s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*DiracDelta(np.sin(b1)) - 2*s1ry*(2 - s1e)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 4*s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))/2)*np.sqrt((s2rx*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2)/np.sqrt((s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2) + (s2rx*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(s1rx*(2 - s1e)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*DiracDelta(np.sin(b1)) - s1ry*(2 - s1e)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1))) + (s2rx*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(s1rx*(2 - s1e)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*DiracDelta(np.sin(b1)) - s1ry*(2 - s1e)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))
        res_ref[1,1] = ((s2rx*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(2*s2rx*(2 - s2e)*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) + 4*s2rx*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*DiracDelta(np.sin(b2)) - 2*s2ry*(2 - s2e)*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 4*s2ry*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2)))/2 + (s2rx*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(2*s2rx*(2 - s2e)*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) + 4*s2rx*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*DiracDelta(np.sin(b2)) - 2*s2ry*(2 - s2e)*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 4*s2ry*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2)))/2)*np.sqrt((s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2)/np.sqrt((s2rx*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2)) + s2ry*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2) + (s1rx*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s2rx*(2 - s2e)*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) + 2*s2rx*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*DiracDelta(np.sin(b2)) - s2ry*(2 - s2e)*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 2*s2ry*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2))) + (s1rx*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1)) + s1ry*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s2rx*(2 - s2e)*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) + 2*s2rx*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*DiracDelta(np.sin(b2)) - s2ry*(2 - s2e)*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 2*s2ry*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2)))
        print("grad F : res = ",res," res_ref = ",res_ref," ecart = ",np.linalg.norm(res-res_ref))
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


    def create_binit(theta_g, theta_d, n, s, liste_pts=None):
        if(liste_pts == None):
            liste_pts = []
        if n == 0:
            return liste_pts
        d1 = np.sqrt(2*(1-np.cos(theta_g-theta_d)))
        xnew = s.rx*( np.sin(theta_g)-np.sin(theta_d) )/d1
        ynew = -s.ry*( np.cos(theta_g)-np.cos(theta_d) )/d1
        xnew2 = xnew*s.rx*s.ry/( ( (s.ry*xnew)**(2/s.e)+(s.rx*ynew)**(2/s.e) )**(s.e/2) )
        ynew2 = ynew*s.rx*s.ry/( ( (s.ry*xnew)**(2/s.e)+(s.rx*ynew)**(2/s.e) )**(s.e/2) )
        sinb = np.sqrt(((ynew2/s.ry)**2)**(1/s.e))
        b = np.arcsin(sinb)
        liste_pts.append(b)
        liste_pts.append(b+np.pi/2)
        liste_pts.append(b+np.pi)
        liste_pts.append(b+3*np.pi/2)
        theta_milieu = 0.5*(theta_g+theta_d)
        create_binit(theta_milieu, theta_d, n-1, s, liste_pts=liste_pts)
        create_binit(theta_g, theta_milieu, n-1, s, liste_pts=liste_pts)
        return liste_pts

    def create_pts(theta_g, theta_d, n, s, liste_pts=None):
        if(liste_pts == None):
            liste_pts = []
        if n == 0:
            return liste_pts
        d1 = np.sqrt(2*(1-np.cos(theta_g-theta_d)))
        xnew = s.rx*( np.sin(theta_g)-np.sin(theta_d) )/d1
        ynew = -s.ry*( np.cos(theta_g)-np.cos(theta_d) )/d1
        # liste_pts.append( s.Mt()@s.Mr()@[xnew, ynew, 0, 1] )
        xnew2 = xnew*s.rx*s.ry/( ( (s.ry*xnew)**(2/s.e)+(s.rx*ynew)**(2/s.e) )**(s.e/2) )
        ynew2 = ynew*s.rx*s.ry/( ( (s.ry*xnew)**(2/s.e)+(s.rx*ynew)**(2/s.e) )**(s.e/2) )

        # cosb = np.sqrt(((xnew2/s.rx)**2)**(1/s.e))
        sinb = np.sqrt(((ynew2/s.ry)**2)**(1/s.e))
        b1 = np.arcsin(sinb)
        b2 = np.arcsin(-sinb)
        # b3 = np.arccos(cosb)
        # b4 = np.arccos(-cosb)
        # print("cosb =",cosb," sinb =",sinb)
        # print("b1 =",b1," b2 =",b2," b3 =",b3," b4 =",b4)
        # print("b1 =",b1," b2 =",b2)
        # pts3 = s.surface_pt(np.array([b1,b2,np.pi-b1,np.pi-b2]))
        pts3 = s.surface_pt(np.array([b1, b1+np.pi/2, b1+np.pi, b1+3*np.pi/2]))# b1, b1+np.pi/2, b1+np.pi, b1-np.pi/2]))
        # print("pts3=",pts3)
        # liste_pts.append( s.Mt()@s.Mr()@[xnew2, ynew2, 0, 1] )
        liste_pts.append( [pts3[0,0], pts3[0,1], 0, 1] )
        liste_pts.append( [pts3[1,0], pts3[1,1], 0, 1] )
        liste_pts.append( [pts3[2,0], pts3[2,1], 0, 1] )
        liste_pts.append( [pts3[3,0], pts3[3,1], 0, 1] )

        theta_milieu = 0.5*(theta_g+theta_d)
        create_pts(theta_milieu, theta_d, n-1, s, liste_pts=liste_pts)
        create_pts(theta_g, theta_milieu, n-1, s, liste_pts=liste_pts)
        return liste_pts


    def test_initialisation():
        s = SuperEllipsoid2D(1,-2,   8,2, e=0.2, theta=-np.pi/5)
        p = pv.Plotter(window_size=[2400,1350])
        L = 10
        grid = pv.UniformGrid()
        arr = np.arange((2*L)**2).reshape((2*L,2*L,1))
        grid.dimensions = np.array(arr.shape) + 1 #dim + 1 because cells
        grid.origin = (-L, -L, 0)
        grid.spacing = (1, 1, 0)
        p.add_mesh(grid, show_edges=True, opacity=0.1)
        p.add_mesh(s.mesh(),color="red", opacity=0.5, name="s")

        theta1 = 0
        theta2 = np.pi/2

        binit = np.array( create_binit(np.pi/2, 0, 5, s, liste_pts=[0, np.pi/2, np.pi, 3*np.pi/2]) )
        # binit = np.array( create_binit(np.pi/2, 0, 4, s) )
        print("binit = ",binit)
        pts = s.surface_pt(binit)
        normals = s.surface_normal(binit)
        # sys.exit()
        # pts = np.array( create_pts(np.pi/2, 0, 5, s, #)[:,:3]
        #     liste_pts=[
        #         s.Mt()@s.Mr()@np.array([s.rx*np.cos(theta2),s.ry*np.sin(theta2),0,1]),
        #         s.Mt()@s.Mr()@np.array([s.rx*np.cos(-theta2),s.ry*np.sin(-theta2),0,1])
        #         # s.Mt()@s.Mr()@np.array([s.rx*np.cos(theta1),s.ry*np.sin(theta1),0,1])
        #     ]) )[:,:3]
        # print("pts =",pts)
        nodes = pv.PolyData(pts)
        nodes["normal"] = normals
        p.add_mesh(nodes.glyph(factor=0.1, geom=pv.Sphere()),color="blue", name="pts")
        p.add_mesh(nodes.glyph(orient="normal",factor=0.25, geom=pv.Arrow()),color="blue", name="normals")

        num = 17
        angles = np.linspace(theta1,theta2,num)
        x = s.rx*np.cos(angles)
        y = s.ry*np.sin(angles)
        pts2 = np.zeros((num,4))
        pts2[:,0] = x
        pts2[:,1] = y
        pts2[:,3] = 1
        for i,pt in enumerate(pts2):
            pts2[i,:] = s.Mt()@s.Mr()@pt
        nodes2 = pv.PolyData(pts2[:,:3])
        # p.add_mesh(nodes2.glyph(factor=0.15, geom=pv.Sphere()),color="green", name="pts2", opacity=0.5)

        pts3 = s.surface_pt(angles)
        nodes3 = pv.PolyData(pts3)
        #p.add_mesh(nodes3.glyph(factor=0.15, geom=pv.Sphere()),color="yellow", name="pts3", opacity=0.5)

        p.show(cpos="xy")

    # test_initialisation()
    # sys.exit()

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
        # binit = np.array([-np.pi, -np.pi/2, 0.001, np.pi/2])
        # pts_ext_s1 = s1.surface_pt(binit)
        # pts_ext_s2 = s2.surface_pt(binit)
        # distances = np.zeros((binit.shape[0],binit.shape[0]))
        # for i in range(binit.shape[0]):
        #     for j in range(binit.shape[0]):
        #         distances[i,j] = np.linalg.norm(pts_ext_s1[i,:]-pts_ext_s2[j,:])
        # # print("distances = ",distances)
        # indmin = np.where(distances==distances.min())
        # u0 = np.array( [ binit[indmin[0][0]], binit[indmin[1][0]] ])
        # print("u0 = ",u0)
        binit1 = np.array( create_binit(np.pi/2, 0, 5, s1, liste_pts=[0, np.pi/2, np.pi, 3*np.pi/2]) )
        pts_ext_s1 = s1.surface_pt(binit1)
        binit2 = np.array( create_binit(np.pi/2, 0, 5, s2, liste_pts=[0, np.pi/2, np.pi, 3*np.pi/2]) )
        pts_ext_s2 = s2.surface_pt(binit2)
        distances = np.zeros((binit1.shape[0],binit2.shape[0]))
        for i in range(binit1.shape[0]):
            for j in range(binit2.shape[0]):
                distances[i,j] = np.linalg.norm(pts_ext_s1[i,:]-pts_ext_s2[j,:])
        indmin = np.where(distances==distances.min())
        u0 = np.array( [ binit1[indmin[0][0]], binit2[indmin[1][0]] ])
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
