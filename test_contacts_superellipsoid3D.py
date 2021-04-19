import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyvista as pv
import scipy
from scipy.optimize import fsolve

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
        ## Vecteurs tangents au solide dans le repere global
        a = sympy.symbols("a",real=True)
        b = sympy.symbols("b",real=True)
        # sympy.ask(sympy.Q.is_true(b>-sympy.pi/2),sympy.Q.is_true(b<sympy.pi/2))
        e = sympy.symbols("e",real=True)
        n = sympy.symbols("n",real=True)
        xt = rx*sign(cos(a))*sympy.Abs(cos(a))**n*sign(cos(b))*sympy.Abs(cos(b))**e
        yt = ry*sign(cos(a))*sympy.Abs(cos(a))**n*sign(sin(b))*sympy.Abs(sin(b))**e
        zt = rz*sign(sin(a))*sympy.Abs(sin(a))**n
        dxtdb = sympy.simplify(sympy.diff(xt,b))
        dytdb = sympy.simplify(sympy.diff(yt,b))
        dztdb = sympy.simplify(sympy.diff(zt,b))
        print("\ncalcul formel : vecteur tangent d (x(a,b),y(a,b),z(a,b)) / db = \n",dxtdb,",\n",dytdb,",\n",dztdb)
        dxtda = sympy.simplify(sympy.diff(xt,a))
        dytda = sympy.simplify(sympy.diff(yt,a))
        dztda = sympy.simplify(sympy.diff(zt,a))
        print("\ncalcul formel : vecteur tangent d (x(a,b),y(a,b),z(a,b)) / da = \n",dxtda,",\n",dytda,",\n",dztda)
        ## Normale exterieure au solide dans le repere global
        normal = Mr@sympy.Matrix([nx,ny,nz,1])
        print("\ncalcul formel : vecteur normal repere global ([nx,ny,nz] dans le repere du solide source) = \n",normal)
        ## Pour la methode de newton
        a1,b1,a2,b2 = sympy.symbols("a1,b1,a2,b2",real=True)
        # superellipsoid1
        s1e = sympy.symbols("s1e",real=True)
        s1n = sympy.symbols("s1n",real=True)
        s1q0,s1q1,s1q2,s1q3 = sympy.symbols("s1q0,s1q1,s1q2,s1q3",real=True)
        s1Mr = sympy.Matrix([[ s1q0**2+s1q1**2-s1q2**2-s1q3**2, 2*s1q1*s1q2-2*s1q0*s1q3, 2*s1q1*s1q3+2*s1q0*s1q2, 0 ],
                           [ 2*s1q1*s1q2+2*s1q0*s1q3, s1q0**2-s1q1**2+s1q2**2-s1q3**2, 2*s1q2*s1q3-2*s1q0*s1q1, 0 ],
                           [ 2*s1q1*s1q3-2*s1q0*s1q2, 2*s1q2*s1q3+2*s1q0*s1q1, s1q0**2-s1q1**2-s1q2**2+s1q3**2, 0 ],
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
        # superellipsoid2
        s2e = sympy.symbols("s2e",real=True)
        s2n = sympy.symbols("s2n",real=True)
        s2q0,s2q1,s2q2,s2q3 = sympy.symbols("s2q0,s2q1,s2q2,s2q3",real=True)
        s2Mr = sympy.Matrix([[ s2q0**2+s2q1**2-s2q2**2-s2q3**2, 2*s2q1*s2q2-2*s2q0*s2q3, 2*s2q1*s2q3+2*s2q0*s2q2, 0 ],
                           [ 2*s2q1*s2q2+2*s2q0*s2q3, s2q0**2-s2q1**2+s2q2**2-s2q3**2, 2*s2q2*s2q3-2*s2q0*s2q1, 0 ],
                           [ 2*s2q1*s2q3-2*s2q0*s2q2, 2*s2q2*s2q3+2*s2q0*s2q1, s2q0**2-s2q1**2-s2q2**2+s2q3**2, 0 ],
                           [ 0,0,0,1]])
        s2rx,s2ry,s2rz,s2xc,s2yc,s2zc = sympy.symbols("s2rx,s2ry,s2rz,s2xc,s2yc,s2zc",real=True)
        s2Ms= sympy.Matrix([[s2rx,0,0,0],[0,s2ry,0,0],[0,0,s2rz,0],[0,0,0,1]])
        s2Mt = sympy.Matrix([[1,0,0,s2xc],[0,1,0,s2yc],[0,0,1,s2zc],[0,0,0,1]])
        s2M = s2Mt@s2Mr@s2Ms
        s2x = sign(cos(a2))*sympy.Abs(cos(a2))**s2n*sign(cos(b2))*sympy.Abs(cos(b2))**s2e
        s2y = sign(cos(a2))*sympy.Abs(cos(a2))**s2n*sign(sin(b2))*sympy.Abs(sin(b2))**s2e
        s2z = sign(sin(a2))*sympy.Abs(sin(a2))**s2n
        pt2 = s2M@sympy.Matrix([s2x,s2y,s2z,1])
        s2nx = s2ry*s2rz*sympy.Abs(cos(a2))**(2-s2n)*sign(cos(b2))*sympy.Abs(cos(b2))**(2-s2e)
        s2ny = s2rx*s2rz*sympy.Abs(cos(a2))**(2-s2n)*sign(sin(b2))*sympy.Abs(sin(b2))**(2-s2e)
        s2nz = s2rx*s2ry*sign(cos(a2))*sign(sin(a2))*sympy.Abs(sin(a2))**(2-s2n)
        n2 = s2Mr@sympy.Matrix([s2nx,s2ny,s2nz,1])
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

    # s1 = SuperEllipsoid3D(0,0,0, 1,1,1, n=1, e=0.5, theta=80, w=[0,0,1])
    # s2 = SuperEllipsoid3D(2,2,3, 1,3,2, n=0.6, e=1, theta=80, w = [1,1,1])
    s1 = SuperEllipsoid3D(0,0,0, 2,1,1, n=0.9, e=0.8, theta=65, w=[0,-1,1])
    s2 = SuperEllipsoid3D(5,4,4, 1,1,1, n=0.4, e=0.9, theta=14, w = [1,1,1])
    # s1.calcul_formel()
    # sys.exit()
    L = 10

    p = pv.Plotter()

    grid = pv.UniformGrid()
    arr = np.arange((2*L)**3).reshape((2*L,2*L,2*L))
    grid.dimensions = np.array(arr.shape) + 1 #dim + 1 because cells
    grid.origin = (-L, -L, -L)
    grid.spacing = (1, 1, 1)
    p.add_mesh(grid, show_edges=True, opacity=0.1)

    p.add_mesh(s1.mesh(),color="red", opacity=0.5)
    p.add_mesh(s2.mesh(),color="green", opacity=0.5)

    aa = np.linspace(-np.pi/2,np.pi/2,num=20)
    bb = np.linspace(-np.pi,np.pi,num=20)
    aaaa, bbbb = np.meshgrid(aa,bb)
    aa = aaaa.reshape((-1,))
    bb = bbbb.reshape((-1,))
    # sys.exit()
    # aa = np.array([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
    # bb = np.array([-np.pi,   -np.pi/2, 0, np.pi/2])
    pts1 = s1.surface_pt(aa,bb)
    normals1 = s1.surface_normal(aa,bb)
    tgt11,tgt12 = s1.surface_tangent(aa,bb)
    # cross1 = np.cross(tgt11,tgt12)
    # cross1[:,0] = cross1[:,0] / np.linalg.norm(cross1,axis=1)
    # cross1[:,1] = cross1[:,1] / np.linalg.norm(cross1,axis=1)
    # cross1[:,2] = cross1[:,2] / np.linalg.norm(cross1,axis=1)
    # print("test cross product : tgt1 x tgt2 / ||tgt1 x tgt2|| - n = ", np.linalg.norm(cross1-normals1,axis=1))
    # sys.exit()
    nodes1 = pv.PolyData(pts1)
    nodes1["normal"] = normals1
    nodes1["tangent1"] = tgt11
    nodes1["tangent2"] = tgt12
    p.add_mesh(nodes1.glyph(factor=0.05, geom=pv.Sphere()),color="blue")
    p.add_mesh(nodes1.glyph(orient="normal",factor=0.25, geom=pv.Arrow()),color="blue")
    p.add_mesh(nodes1.glyph(orient="tangent1",factor=0.25, geom=pv.Arrow()),color="magenta")
    p.add_mesh(nodes1.glyph(orient="tangent2",factor=0.25, geom=pv.Arrow()),color="magenta")

    pts2 = s2.surface_pt(aa,bb)
    normals2 = s2.surface_normal(aa,bb)
    tgt21,tgt22 = s2.surface_tangent(aa,bb)
    nodes2 = pv.PolyData(pts2)
    nodes2["normal"] = normals2
    nodes2["tangent1"] = tgt21
    nodes2["tangent2"] = tgt22
    p.add_mesh(nodes2.glyph(factor=0.05, geom=pv.Sphere()),color="blue")
    p.add_mesh(nodes2.glyph(orient="normal",factor=0.25, geom=pv.Arrow()),color="blue")
    p.add_mesh(nodes2.glyph(orient="tangent1",factor=0.25, geom=pv.Arrow()),color="magenta")
    p.add_mesh(nodes2.glyph(orient="tangent2",factor=0.25, geom=pv.Arrow()),color="magenta")

    # # Calcul de la distance entre deux superellipsoides
    # # (1) appartenance à la superellipsoide 1 => b1 => x1=x(b1), y1=y(b1)
    # # (2) appartenance à la superellipsoide 2 => b2 => x2=x(b2), y2=y(b2)
    # # (3) soient X1=(x1,y1) X2=(x2,y2)
    # #     on veut X2X1 ^ n1 = 0 (i.e. X2X1 et n1 colineaires)
    # #     (x2-x1)   nx1         (y2-y1)*nz1 - (z2-z1)*ny1 = 0
    # #     (y2-y1)   ny1    i.e. (z2-z1)*nx1 - (x2-x1)*nz1 = 0
    # #     (z2-z1)   nz1         (x2-x1)*ny1 - (y2-y1)*nx1 = 0
    # #     (x2-x1)   nx1
    # #     (y2-y1)   ny1
    # # (4) normales opposées : n1.n2 + ||n1||*||n2|| = 0 (i.e. n1=-n2)
    #
    def f_contacts(u,s1,s2):
        a1,b1,a2,b2 = u
        ## version 1 (2D)
        # X1 = s1.surface_pt(np.array([b1]))
        # X2 = s2.surface_pt(np.array([b2]))
        # N1 = s1.surface_normal(np.array([b1]))
        # N2 = s2.surface_normal(np.array([b2]))
        # res = np.zeros((2,))
        # # res[0] = (X2[0,0]-X1[0,0])*N1[0,1] -  (X2[0,1]-X1[0,1])*N1[0,0]
        # res[0] = -(X2[0,0]-X1[0,0])*N2[0,1] +  (X2[0,1]-X1[0,1])*N2[0,0]
        # # res[0] = np.abs((X2[0,0]-X1[0,0])*N1[0,1] -  (X2[0,1]-X1[0,1])*N1[0,0])+ np.abs(-(X2[0,0]-X1[0,0])*N2[0,1] +  (X2[0,1]-X1[0,1])*N2[0,0])
        # res[1] = np.dot(N1[0,:],N2[0,:]) + np.linalg.norm(N1[0,:])*np.linalg.norm(N1[0,:])
        ## version 2 (sympy)
        res = np.zeros((4,))
        s1q0,s1q1,s1q2,s1q3 = s1.q
        s1n = s1.n ; s1e = s1.e ; s1rx = s1.rx ; s1ry = s1.ry ; s1rz = s1.rz ; s1xc = s1.xc ; s1yc = s1.yc ; s1zc = s1.zc
        s2q0,s2q1,s2q2,s2q3 = s2.q
        s2n = s2.n ; s2e = s2.e ; s2rx = s2.rx ; s2ry = s2.ry ; s2rz = s2.rz ; s2xc = s2.xc ; s2yc = s2.yc ; s2zc = s2.zc

        ## res_ref est directement obtenu a partir du calcul formel
        ## permet de tester que la reecriture des formules n'a pas fait ajouter des erreurs
        res_ref = np.zeros((4,))
        res_ref[0] = -(s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1zc + s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2zc) + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1yc + s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2yc)

        res_ref[1] = (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1zc + s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2zc) - (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1xc + s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2xc)

        res_ref[2] = (s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1xc + s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2xc) - (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1yc + s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2yc)

        res_ref[3] = (s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))) + (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))) + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))) + np.sqrt((s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2)*np.sqrt((s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2)


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
        Q1 = (s1q00 + s1q11 - s1q22 - s1q33)   # M1[0,0]
        R1 = (s2q00 + s2q11 - s2q22 - s2q33)   # M2[0,0]
        Q2 = (s1q00 - s1q11 + s1q22 - s1q33)   # M1[1,1]
        R2 = (s2q00 - s2q11 + s2q22 - s2q33)   # M2[1,1]
        Q3 = (s1q03 + s1q12) # M1[1,0]
        R3 = (s2q03 + s2q12) # M2[1,0]
        Q4 = (-s1q03 + s1q12) # M1[0,1]
        R4 = (-s2q03 + s2q12) # M2[0,1]
        Q5 = (-2*s1q0*s1q1 + 2*s1q2*s1q3)  # M1[1,2]
        R5 = (-2*s2q0*s2q1 + 2*s2q2*s2q3)  # M2[1,2]
        Q6 = (-2*s1q0*s1q2 + 2*s1q1*s1q3)  # M1[2,0]
        R6 = (-2*s2q0*s2q2 + 2*s2q1*s2q3)  # M2[2,0]
        Q7 = (2*s1q0*s1q1 + 2*s1q2*s1q3)   # M1[2,1]
        R7 = (2*s2q0*s2q1 + 2*s2q2*s2q3)   # M2[2,1]
        Q8 = (s1q00 - s1q11 - s1q22 + s1q33)   # M1[2,2]
        R8 = (s2q00 - s2q11 - s2q22 + s2q33)   # M2[2,2]
        Q9 = (2*s1q0*s1q2 + 2*s1q1*s1q3)    # M1[0,2]
        R9 = (2*s2q0*s2q2 + 2*s2q1*s2q3)    # M2[0,2]

        # CA1 = np.abs(np.cos(a1))**s1n*np.sign(np.cos(a1))
        # SA1 = np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))
        #
        # CA1N1 = np.abs(np.cos(a1))**(1-s1n)*np.sign(np.cos(a1))
        # SA1N1 = np.abs(np.sin(a1))**(1-s1n)*np.sign(np.sin(a1))
        #
        # SA1N2 = np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))
        #
        # CA2 = np.abs(np.cos(a2))**s2n*np.sign(np.cos(a2))
        # SA2 = np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))
        #
        # SA2N2 = np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))
        #
        # CB1 = np.abs(np.cos(b1))**s1e*np.sign(np.cos(b1))
        # CB1E2 = np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))
        #
        # SB1 = np.abs(np.sin(b1))**s1e*np.sign(np.sin(b1))
        # SB1E2 = np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1))
        #
        # CB2 = np.abs(np.cos(b2))**s2e*np.sign(np.cos(b2))
        # CB2E2 = np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))
        #
        # SB2 = np.abs(np.sin(b2))**s2e*np.sign(np.sin(b2))
        # SB2E2 = np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2))
        #
        # H01 = SA1N2*np.sign(np.cos(a1))
        # H02 = CA1*CB1
        # H03 = SB1E2*np.abs(np.cos(a1))**(2 - s1n)
        # H04  = np.abs(np.cos(a1))**(2 - s1n)*CB1E2
        #
        # H6 = np.sin(a1)*np.abs(np.cos(a1))**(s1n-1)*CB1
        # H7 = np.sin(a1)*SB1*np.abs(np.cos(a1))**(s1n-1)
        # H8 = np.cos(b1)*np.abs(np.sin(b1))**(1 - s1e)*np.abs(np.cos(a1))**(2 - s1n)
        # H9 = SB1*CA1
        # H10 = CA2*CB2
        # H11 = SB2*CA2
        # H12 = np.sin(a1)*SB1E2*CA1N1
        # H13 = np.sin(a1)*CA1N1*CB1E2
        # H14 = np.cos(a1)*np.abs(np.sin(a1))**(1 - s1n)*np.sign(np.cos(a1))
        # H15 = np.cos(a1)*np.abs(np.sin(a1))**(s1n-1)
        # H16 = np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(1 - s1e)
        # H17 = SB2E2*np.abs(np.cos(a2))**(2 - s2n)
        # H18 = SA2N2*np.sign(np.cos(a2))
        # H19 = np.abs(np.cos(a2))**(2 - s2n)
        # H20 = np.sin(b1)*CA1*np.abs(np.cos(b1))**(s1e-1)
        # H21 = np.cos(b1)*np.abs(np.sin(b1))**(s1e-1)
        # H22 = np.sin(a2)*SB2*np.abs(np.cos(a2))**(s2n-1)
        # H23 = np.sin(a2)*np.abs(np.cos(a2))**(s2n-1)
        # H24 = np.sin(b2)*CA2*np.abs(np.cos(b2))**(s2e-1)
        # H25 = np.cos(b2)*np.abs(np.sin(b2))**(s2e-1)
        # H26 = np.cos(a2)*np.abs(np.sin(a2))**(s2n-1)
        # H27 = np.cos(a2)*np.sign(np.cos(a2))*np.abs(np.sin(a2))**(1 - s2n)
        # H28 = np.cos(b2)*np.abs(np.sin(b2))**(1 - s2e)
        # H29 = np.sin(a2)*SB2E2*np.abs(np.cos(a2))**(1 - s2n)*np.sign(np.cos(a2))
        # H30 = np.sin(b2)*H19*np.abs(np.cos(b2))**(1 - s2e)
        # H31 = np.sin(a2)*np.abs(np.cos(a2))**(1 - s2n)*np.sign(np.cos(a2))*CB2E2
        # H32 = SB2*CA2
        # H33 = H19*CB2E2
        # H34 = H28*H19
        # H35 = H21*CA1
        # H36 = H23*CB2
        # H37 = H25*CA2


        CA1 = np.abs(np.cos(a1))**s1n*np.sign(np.cos(a1))
        CA2 = np.abs(np.cos(a2))**s2n*np.sign(np.cos(a2))
        CB1 = np.abs(np.cos(b1))**s1e*np.sign(np.cos(b1))
        CB2 = np.abs(np.cos(b2))**s2e*np.sign(np.cos(b2))

        SA1 = np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))
        SA2 = np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))
        SB1 = np.abs(np.sin(b1))**s1e*np.sign(np.sin(b1))
        SB2 = np.abs(np.sin(b2))**s2e*np.sign(np.sin(b2))

        #
        # CA1N1 = np.abs(np.cos(a1))**(1 - s1n)*np.sign(np.cos(a1))
        # CA2N1 = np.abs(np.cos(a2))**(1 - s2n)*np.sign(np.cos(a2))
        # CB1E1 = np.abs(np.cos(b1))**(1 - s1e)*np.sign(np.cos(b1))
        # CB2E1 = np.abs(np.cos(b2))**(1 - s2e)*np.sign(np.cos(b2))
        #
        # SA1N1 = np.abs(np.sin(a1))**(1 - s1n)*np.sign(np.sin(a1))
        # SA2N1 = np.abs(np.sin(a2))**(1 - s2n)*np.sign(np.sin(a2))
        # SB1E1 = np.abs(np.sin(b1))**(1 - s1e)*np.sign(np.sin(b1))
        # SB2E1 = np.abs(np.sin(b2))**(1 - s2e)*np.sign(np.sin(b2))
        #
        # CA1N2 = np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.cos(a1))
        # CA2N2 = np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.cos(a2))
        # CB1E2 = np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))
        # CB2E2 = np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))
        #
        # SA1N2 = np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))
        # SA2N2 = np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))
        # SB1E2 = np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1))
        # SB2E2 = np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2))
        #
        # H01 = SA1N2*np.sign(np.cos(a1))
        # H02 = CA1*CB1
        # H03 = SB1E2*CA1N2*np.sign(np.cos(a1))
        # H04 = CB1E2*CA1N2*np.sign(np.cos(a1))
        #
        # #H6 = np.sin(a1)*np.abs(np.cos(a1))**(s1n-1)*CB1
        # H6 = np.sin(a1)*np.sign(np.cos(a1))*CB1/CA1N1
        # # H7 = np.sin(a1)*SB1*np.abs(np.cos(a1))**(s1n-1)
        # H7 = np.sin(a1)*np.sign(np.cos(a1))*SB1/CA1N1
        # # H8 = np.cos(b1)*np.abs(np.sin(b1))**(1 - s1e)*np.abs(np.cos(a1))**(2 - s1n)
        # H8 = np.cos(b1)*SB1E1*CA1N2*np.sign(np.sin(b1))*np.sign(np.cos(a1))
        #
        # H9 = SB1*CA1
        # H10 = CA2*CB2
        # H11 = SB2*CA2
        # H12 = np.sin(a1)*SB1E2*CA1N1
        # H13 = np.sin(a1)*CA1N1*CB1E2
        # H14 = np.abs(np.cos(a1))*SA1N1*np.sign(np.sin(a1))
        #
        # H15 = np.cos(a1)*np.sign(np.sin(a1))/SA1N1
        #
        # H16 = np.sin(b1)*CA1N2*np.sign(np.cos(a1))*CB1E1*np.sign(np.cos(b1))
        # H17 = SB2E2*CA2N2*np.sign(np.cos(a2))
        # H18 = SA2N2*np.sign(np.cos(a2))
        # H19 = CA2N2*np.sign(np.cos(a2))
        # H20 = np.sin(b1)*CA1/CB1E1*np.sign(np.cos(b1))
        # H21 = np.cos(b1)/SB1E1*np.sign(np.sin(b1))
        #
        # H22 = np.sin(a2)*SB2/CA2N1*np.sign(np.cos(a2))
        # H23 = np.sin(a2)/CA2N1*np.sign(np.cos(a2))
        # H24 = np.sin(b2)*CA2/CB2E1*np.sign(np.cos(b2))
        # H25 = np.cos(b2)/SB2E1*np.sign(np.sin(b2))
        # H26 = np.cos(a2)/SA2N1*np.sign(np.sin(a2))
        #
        # H27 = np.abs(np.cos(a2))*SA2N1*np.sign(np.sin(a2))
        # H28 = np.cos(b2)*SB2E1*np.sign(np.sin(b2))
        #
        # H29 = np.sin(a2)*SB2E2*CA2N1
        # H30 = np.sin(b2)*H19*CB2E1*np.sign(np.cos(b2))
        # H31 = np.sin(a2)*CA2N1*CB2E2
        # H32 = SB2*CA2
        # H33 = H19*CB2E2
        # H34 = H28*H19
        # H35 = H21*CA1
        # H36 = H23*CB2
        # H37 = H25*CA2

        H01 = np.sign(np.cos(a1))*np.sin(a1)**2/SA1
        H02 = CA1*CB1
        H03 = np.sin(b1)**2*np.cos(a1)*np.abs(np.cos(a1))/(SB1*CA1)
        H04 = np.cos(b1)**2*np.cos(a1)*np.abs(np.cos(a1))/(CB1*CA1)

        H6 = np.tan(a1)*CB1*CA1

        # H7 = np.sin(a1)*SB1*np.abs(np.cos(a1))**(s1n-1)
        H7 = np.tan(a1)*SB1*CA1
        # H8 = np.cos(b1)*np.abs(np.sin(b1))**(1 - s1e)*np.abs(np.cos(a1))**(2 - s1n)
        H8 = np.cos(b1)*np.sin(b1)*np.cos(a1)*np.abs(np.cos(a1))/(CA1*SB1)

        H9 = SB1*CA1
        H10 = CA2*CB2
        H11 = SB2*CA2
        H12 = np.sin(a1)*np.sin(b1)**2*np.abs(np.cos(a1))/(SB1*CA1)
        H13 = np.sin(a1)*np.abs(np.cos(a1))*np.cos(b1)**2/(CA1*CB1)
        H14 = np.abs(np.cos(a1))*np.sin(a1)/SA1
        H15 = SA1/np.tan(a1)

        H16 = np.sin(b1)*np.cos(b1)*np.cos(a1)*np.abs(np.cos(a1))/(CB1*CA1)
        H17 = np.sin(b2)**2*np.cos(a2)*np.abs(np.cos(a2))/(SB2*CA2)
        H18 = np.sin(a2)**2*np.sign(np.cos(a2))/SA2
        H19 = np.cos(a2)*np.abs(np.cos(a2))/CA2

        # H20 = np.sin(b1)*CA1/CB1E1*np.sign(np.cos(b1))
        H20 = CA1*CB1*np.tan(b1)

        #H21 = np.cos(b1)/SB1E1*np.sign(np.sin(b1))
        H21 = SB1/np.tan(b1)

        # H22 = np.sin(a2)*SB2/CA2N1*np.sign(np.cos(a2))
        H22 = SB2*CA2*np.tan(a2)

        # H23 = np.sin(a2)/CA2N1*np.sign(np.cos(a2))
        H23 = CA2*np.tan(a2)

        # H24 = np.sin(b2)*CA2/CB2E1*np.sign(np.cos(b2))
        H24 = CA2*CB2*np.tan(b2)


        # H25 = np.cos(b2)/SB2E1*np.sign(np.sin(b2))
        H25 = SB2/np.tan(b2)
        # H26 = np.cos(a2)/SA2N1*np.sign(np.sin(a2))
        H26 = SA2/np.tan(a2)

        H27 = np.abs(np.cos(a2))*np.sin(a2)/SA2
        H28 = np.cos(b2)*np.sin(b2)/SB2
        H29 = np.sin(a2)*np.sin(b2)**2*np.abs(np.cos(a2))/(SB2*CA2)
        H30 = np.sin(b2)*np.cos(a2)*np.abs(np.cos(a2))*np.cos(b2)/(CA2*CB2)
        H31 = np.sin(a2)*np.abs(np.cos(a2))*np.cos(b2)**2/(CA2*CB2)
        H32 = SB2*CA2
        H33 = np.cos(a2)*np.abs(np.cos(a2))*np.cos(b2)**2/(CA2*CB2)

        H34 = np.cos(b2)*np.sin(b2)*np.cos(a2)*np.abs(np.cos(a2))/(SB2*CA2)

        # H35 = np.cos(b1)/SB1E1*np.sign(np.sin(b1))*CA1
        H35 = SB1*CA1/np.tan(b1)

        # H36 = np.sin(a2)/CA2N1*np.sign(np.cos(a2))*CB2
        H36 = CA2*CB2*np.tan(a2)

        # H37 = np.cos(b2)/SB2E1*np.sign(np.sin(b2))*CA2
        H37 = SB2*CA2/np.tan(b2)


        res[0] = -(s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04)* \
                  (-s1rx*Q6*H02 - s1ry*Q7*H9 - s1rz*Q8*SA1 - s1zc + s2rx*R6*H10 + s2ry*R7*H11 + s2rz*R8*SA2 + s2zc) + \
                  (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04)* \
                  (-s1rx*Q3*H02 - s1ry*Q2*H9 - s1rz*Q5*SA1 - s1yc + s2rx*R3*H10 + s2ry*R2*H11 + s2rz*R5*SA2 + s2yc)

        res[1] = (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04)* \
                 (-s1rx*Q6*H02 - s1ry*Q7*H9 - s1rz*Q8*SA1 - s1zc + s2rx*R6*H10 + s2ry*R7*H11 + s2rz*R8*SA2 + s2zc) - \
                 (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04)* \
                 (-s1rx*Q1*H02 - s1ry*Q4*H9 - s1rz*Q9*SA1 - s1xc + s2rx*R1*H10 + s2ry*R4*H11 + s2rz*R9*SA2 + s2xc)

        res[2] = (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04)* \
                 (-s1rx*Q1*H02 - s1ry*Q4*H9 - s1rz*Q9*SA1 - s1xc + s2rx*R1*H10 + s2ry*R4*H11 + s2rz*R9*SA2 + s2xc) - \
                 (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04)* \
                 (-s1rx*Q3*H02 - s1ry*Q2*H9 - s1rz*Q5*SA1 - s1yc + s2rx*R3*H10 + s2ry*R2*H11 + s2rz*R5*SA2 + s2yc)

        res[3] = (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04)* \
                 (s2rx*s2ry*R5*H18 + s2rx*s2rz*R2*H17 + s2ry*s2rz*R3*H33) + \
                 (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04)* \
                 (s2rx*s2ry*R9*H18 + s2rx*s2rz*R4*H17 + s2ry*s2rz*R1*H33) + \
                 (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04)* \
                 (s2rx*s2ry*R8*H18 + s2rx*s2rz*R7*H17 + s2ry*s2rz*R6*H33) + \
                 np.sqrt((s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04)**2 + \
                         (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04)**2 + \
                         (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04)**2 \
                        )* \
                np.sqrt((s2rx*s2ry*R5*H18 + s2rx*s2rz*R2*H17 + s2ry*s2rz*R3*H33)**2 + \
                        (s2rx*s2ry*R9*H18 + s2rx*s2rz*R4*H17 + s2ry*s2rz*R1*H33)**2 + \
                        (s2rx*s2ry*R8*H18 + s2rx*s2rz*R7*H17 + s2ry*s2rz*R6*H33)**2 \
                       )

        print("F : res = ",res," res_ref = ",res_ref," ecart = ",np.linalg.norm(res-res_ref))

        return res

    def DiracDelta(x):
        return 0

    def grad_f_contacts(u,s1,s2):
        res = np.zeros((4,4))
        a1,b1,a2,b2 = u
        s1q0,s1q1,s1q2,s1q3 = s1.q
        s1n = s1.n ; s1e = s1.e ; s1rx = s1.rx ; s1ry = s1.ry ; s1rz = s1.rz ; s1xc = s1.xc ; s1yc = s1.yc ; s1zc = s1.zc
        s2q0,s2q1,s2q2,s2q3 = s2.q
        s2n = s2.n ; s2e = s2.e ; s2rx = s2.rx ; s2ry = s2.ry ; s2rz = s2.rz ; s2xc = s2.xc ; s2yc = s2.yc ; s2zc = s2.zc

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
        R1 = (s2q00 + s2q11 - s2q22 - s2q33)
        Q2 = (s1q00 - s1q11 + s1q22 - s1q33)
        R2 = (s2q00 - s2q11 + s2q22 - s2q33)
        Q3 = (s1q03 + s1q12)
        R3 = (s2q03 + s2q12)
        Q4 = (-s1q03 + s1q12)
        R4 = (-s2q03 + s2q12)
        Q5 = (-2*s1q0*s1q1 + 2*s1q2*s1q3)
        R5 = (-2*s2q0*s2q1 + 2*s2q2*s2q3)
        Q6 = (-2*s1q0*s1q2 + 2*s1q1*s1q3)
        R6 = (-2*s2q0*s2q2 + 2*s2q1*s2q3)
        Q7 = (2*s1q0*s1q1 + 2*s1q2*s1q3)
        R7 = (2*s2q0*s2q1 + 2*s2q2*s2q3)
        Q8 = (s1q00 - s1q11 - s1q22 + s1q33)
        R8 = (s2q00 - s2q11 - s2q22 + s2q33)
        Q9 = (2*s1q0*s1q2 + 2*s1q1*s1q3)
        R9 = (2*s2q0*s2q2 + 2*s2q1*s2q3)

        CA1 = np.abs(np.cos(a1))**s1n*np.sign(np.cos(a1))
        CA2 = np.abs(np.cos(a2))**s2n*np.sign(np.cos(a2))
        CB1 = np.abs(np.cos(b1))**s1e*np.sign(np.cos(b1))
        CB2 = np.abs(np.cos(b2))**s2e*np.sign(np.cos(b2))

        SA1 = np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))
        SA2 = np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))
        SB1 = np.abs(np.sin(b1))**s1e*np.sign(np.sin(b1))
        SB2 = np.abs(np.sin(b2))**s2e*np.sign(np.sin(b2))

        # u =  [-0.72414491 -1.09268482 -0.76188784 -2.5141922 ]   [-0.72414442 -1.0926831  -0.76189723 -2.51419776]

        # CA1N1 = np.abs(np.cos(a1))/CA1 # np.abs(np.cos(a1))**(1 - s1n)*np.sign(np.cos(a1))
        # CA2N1 = np.abs(np.cos(a2))/CA2 # np.abs(np.cos(a2))**(1 - s2n)*np.sign(np.cos(a2))
        # CB1E1 = np.abs(np.cos(b1))/CB1 # np.abs(np.cos(b1))**(1 - s1e)*np.sign(np.cos(b1))
        # CB2E1 = np.abs(np.cos(b2))/CB2 # np.abs(np.cos(b2))**(1 - s2e)*np.sign(np.cos(b2))
        #
        # SA1N1 = np.abs(np.sin(a1))/SA1 # np.abs(np.sin(a1))**(1 - s1n)*np.sign(np.sin(a1))
        # SA2N1 = np.abs(np.sin(a2))/SA2 # np.abs(np.sin(a2))**(1 - s2n)*np.sign(np.sin(a2))
        # SB1E1 = np.abs(np.sin(b1))/SB1 # np.abs(np.sin(b1))**(1 - s1e)*np.sign(np.sin(b1))
        # SB2E1 = np.abs(np.sin(b2))/SB2 # np.abs(np.sin(b2))**(1 - s2e)*np.sign(np.sin(b2))
        #
        # CA1N2 = np.cos(a1)**2/CA1 # np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.cos(a1))
        # CA2N2 = np.cos(a2)**2/CA2 # np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.cos(a2))
        # CB1E2 = np.cos(b1)**2/CB1 # np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))
        # CB2E2 = np.cos(b2)**2/CB2 # np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))
        #
        # SA1N2 = np.sin(a1)**2/SA1 # np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))
        # SA2N2 = np.sin(a2)**2/SA2 # np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))
        # SB1E2 = np.sin(b1)**2/SB1 # np.abs(np.sin(b1))**(2 - s1e)*np.sign(np.sin(b1))
        # SB2E2 = np.sin(b2)**2/SB2 # np.abs(np.sin(b2))**(2 - s2e)*np.sign(np.sin(b2))

        # H01 = np.sign(np.cos(a1))*np.sin(a1)**2/SA1
        H01 = np.sign(np.cos(a1))*np.abs(np.sin(a1))**(2-s1n)*np.sign(np.sin(a1))
        H02 = CA1*CB1
        # H03 = np.sin(b1)**2*np.cos(a1)*np.abs(np.cos(a1))/(SB1*CA1)
        H03 = np.sign(np.sin(b1))*np.abs(np.sin(b1))**(2-s1e)*np.abs(np.cos(a1))**(2-s1n)

        # H04 = np.cos(b1)**2*np.cos(a1)*np.abs(np.cos(a1))/(CB1*CA1)
        H04 = np.sign(np.cos(b1))*np.abs(np.cos(b1))**(2-s1e)*np.abs(np.cos(a1))**(2-s1n)

        H6 = np.tan(a1)*CB1*CA1

        # H7 = np.sin(a1)*SB1*np.abs(np.cos(a1))**(s1n-1)
        H7 = np.tan(a1)*SB1*CA1
        # H8 = np.cos(b1)*np.abs(np.sin(b1))**(1 - s1e)*np.abs(np.cos(a1))**(2 - s1n)
        H8 = np.cos(b1)*np.sin(b1)*np.cos(a1)*np.abs(np.cos(a1))/(CA1*SB1)

        H9  = CA1*SB1
        H10 = CA2*CB2
        H11 = CA2*SB2
        H12 = np.sin(a1)*np.sin(b1)**2*np.abs(np.cos(a1))/(SB1*CA1)
        H13 = np.sin(a1)*np.abs(np.cos(a1))*np.cos(b1)**2/(CA1*CB1)
        H14 = np.abs(np.cos(a1))*np.sin(a1)/SA1
        H15 = SA1/np.tan(a1)

        H16 = np.sin(b1)*np.cos(b1)*np.cos(a1)*np.abs(np.cos(a1))/(CB1*CA1)
        H17 = np.sin(b2)**2*np.cos(a2)*np.abs(np.cos(a2))/(SB2*CA2)
        H18 = np.sin(a2)**2*np.sign(np.cos(a2))/SA2
        H19 = np.cos(a2)*np.abs(np.cos(a2))/CA2

        # H20 = np.sin(b1)*CA1/CB1E1*np.sign(np.cos(b1))
        H20 = CA1*CB1*np.tan(b1)

        #H21 = np.cos(b1)/SB1E1*np.sign(np.sin(b1))
        H21 = SB1/np.tan(b1)

        # H22 = np.sin(a2)*SB2/CA2N1*np.sign(np.cos(a2))
        H22 = SB2*CA2*np.tan(a2)

        # H23 = np.sin(a2)/CA2N1*np.sign(np.cos(a2))
        H23 = CA2*np.tan(a2)

        # H24 = np.sin(b2)*CA2/CB2E1*np.sign(np.cos(b2))
        H24 = CA2*CB2*np.tan(b2)


        # H25 = np.cos(b2)/SB2E1*np.sign(np.sin(b2))
        H25 = SB2/np.tan(b2)
        # H26 = np.cos(a2)/SA2N1*np.sign(np.sin(a2))
        H26 = SA2/np.tan(a2)

        H27 = np.abs(np.cos(a2))*np.sin(a2)/SA2
        H28 = np.cos(b2)*np.sin(b2)/SB2
        H29 = np.sin(a2)*np.sin(b2)**2*np.abs(np.cos(a2))/(SB2*CA2)
        H30 = np.sin(b2)*np.cos(a2)*np.abs(np.cos(a2))*np.cos(b2)/(CA2*CB2)
        H31 = np.sin(a2)*np.abs(np.cos(a2))*np.cos(b2)**2/(CA2*CB2)
        H32 = SB2*CA2
        H33 = np.cos(a2)*np.abs(np.cos(a2))*np.cos(b2)**2/(CA2*CB2)

        H34 = np.cos(b2)*np.sin(b2)*np.cos(a2)*np.abs(np.cos(a2))/(SB2*CA2)

        # H35 = np.cos(b1)/SB1E1*np.sign(np.sin(b1))*CA1
        H35 = SB1*CA1/np.tan(b1)

        # H36 = np.sin(a2)/CA2N1*np.sign(np.cos(a2))*CB2
        H36 = CA2*CB2*np.tan(a2)

        # H37 = np.cos(b2)/SB2E1*np.sign(np.sin(b2))*CA2
        H37 = SB2*CA2/np.tan(b2)

        res[0,0] = (-s1rx*s1ry*Q5*H01 - s1rx*s1rz*Q2*H03 - s1ry*s1rz*Q3*H04)*\
                   (s1n*s1rx*Q6*H6 + s1n*s1ry*Q7*H7 - s1n*s1rz*Q8*H15) + \
                   (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04)*\
                   (s1n*s1rx*Q3*H6 + s1n*s1ry*Q2*H7 - s1n*s1rz*Q5*H15) + \
                   (-s1rx*s1ry*(2 - s1n)*Q5*H14 + s1rx*s1rz*(2 - s1n)*Q2*H12 + s1ry*s1rz*(2 - s1n)*Q3*H13)* \
                   (-s1rx*Q6*H02 - s1ry*Q7*H9 - s1rz*Q8*SA1 - s1zc + s2rx*R6*H10 + s2ry*R7*H11 + s2rz*R8*SA2 + s2zc) + \
                   (s1rx*s1ry*(2 - s1n)*Q8*H14 - s1rx*s1rz*(2 - s1n)*Q7*H12 - s1ry*s1rz*(2 - s1n)*Q6*H13)* \
                   (-s1rx*Q3*H02 - s1ry*Q2*H9 - s1rz*Q5*SA1 - s1yc + s2rx*R3*H10 + s2ry*R2*H11 + s2rz*R5*SA2 + s2yc)

        res[0,1] = (-s1rx*s1ry*Q5*H01 - s1rx*s1rz*Q2*H03 - s1ry*s1rz*Q3*H04)* \
                   (s1e*s1rx*Q6*H20 - s1e*s1ry*Q7*H35) + \
                   (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04)*\
                   (s1e*s1rx*Q3*H20 - s1e*s1ry*Q2*H35) + \
                   (s1rx*s1rz*(2 - s1e)*Q7*H8 - s1ry*s1rz*(2 - s1e)*Q6*H16)* \
                   (-s1rx*Q3*H02 - s1ry*Q2*H9 - s1rz*Q5*SA1 - s1yc + s2rx*R3*H10 + s2ry*R2*H11 + s2rz*R5*SA2 + s2yc) + \
                   (-s1rx*s1rz*(2 - s1e)*Q2*H8 + s1ry*s1rz*(2 - s1e)*Q3*H16)* \
                   (-s1rx*Q6*H02 - s1ry*Q7*H9 - s1rz*Q8*SA1 - s1zc + s2rx*R6*H10 + s2ry*R7*H11 + s2rz*R8*SA2 + s2zc)

        res[0,2] = (-s1rx*s1ry*Q5*H01 - s1rx*s1rz*Q2*H03 - s1ry*s1rz*Q3*H04)* \
                   (-s2n*s2rx*R6*H36 - s2n*s2ry*R7*H22 + s2n*s2rz*R8*H26) + \
                   (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04)*\
                   (-s2n*s2rx*R3*H36 - s2n*s2ry*R2*H22 + s2n*s2rz*R5*H26)

        res[0,3] = (-s1rx*s1ry*Q5*H01 - s1rx*s1rz*Q2*H03 - s1ry*s1rz*Q3*H04)* \
                   (-s2e*s2rx*R6*H24 + s2e*s2ry*R7*H37) + \
                   (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04)*\
                   (-s2e*s2rx*R3*H24 + s2e*s2ry*R2*H37)

        res[1,0] = (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04)* \
                   (s1n*s1rx*Q6*H6 + s1n*s1ry*Q7*H7 - s1n*s1rz*Q8*H15) + \
                   (-s1rx*s1ry*Q8*H01 - s1rx*s1rz*Q7*H03 - s1ry*s1rz*Q6*H04)*\
                   (s1n*s1rx*Q1*H6 + s1n*s1ry*Q4*H7 - s1n*s1rz*Q9*H15) + \
                   (s1rx*s1ry*(2 - s1n)*Q9*H14 - s1rx*s1rz*(2 - s1n)*Q4*H12 - s1ry*s1rz*(2 - s1n)*Q1*H13)* \
                   (-s1rx*Q6*H02 - s1ry*Q7*H9 - s1rz*Q8*SA1 - s1zc + s2rx*R6*H10 + s2ry*R7*H11 + s2rz*R8*SA2 + s2zc) + \
                   (-s1rx*s1ry*(2 - s1n)*Q8*H14 + s1rx*s1rz*(2 - s1n)*Q7*H12 + s1ry*s1rz*(2 - s1n)*Q6*H13)* \
                   (-s1rx*Q1*H02 - s1ry*Q4*H9 - s1rz*Q9*SA1 - s1xc + s2rx*R1*H10 + s2ry*R4*H11 + s2rz*R9*SA2 + s2xc)

        res[1,1] = (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04)* \
                   (s1e*s1rx*Q6*H20 - s1e*s1ry*Q7*H35) + \
                   (-s1rx*s1ry*Q8*H01 - s1rx*s1rz*Q7*H03 - s1ry*s1rz*Q6*H04)* \
                   (s1e*s1rx*Q1*H20 - s1e*s1ry*Q4*H35) + \
                   (-s1rx*s1rz*(2 - s1e)*Q7*H8 + s1ry*s1rz*(2 - s1e)*Q6*H16)* \
                   (-s1rx*Q1*H02 - s1ry*Q4*H9 - s1rz*Q9*SA1 - s1xc + s2rx*R1*H10 + s2ry*R4*H11 + s2rz*R9*SA2 + s2xc) + \
                   (s1rx*s1rz*(2 - s1e)*Q4*H8 - s1ry*s1rz*(2 - s1e)*Q1*H16)* \
                   (-s1rx*Q6*H02 - s1ry*Q7*H9 - s1rz*Q8*SA1 - s1zc + s2rx*R6*H10 + s2ry*R7*H11 + s2rz*R8*SA2 + s2zc)

        res[1,2] = (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04)* \
                   (-s2n*s2rx*R6*H36 - s2n*s2ry*R7*H22 + s2n*s2rz*R8*H26) + \
                   (-s1rx*s1ry*Q8*H01 - s1rx*s1rz*Q7*H03 - s1ry*s1rz*Q6*H04)* \
                   (-s2n*s2rx*R1*H36 - s2n*s2ry*R4*H22 + s2n*s2rz*R9*H26)

        res[1,3] = (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04)* \
                   (-s2e*s2rx*R6*H24 + s2e*s2ry*R7*H37) + \
                   (-s1rx*s1ry*Q8*H01 - s1rx*s1rz*Q7*H03 - s1ry*s1rz*Q6*H04)* \
                   (-s2e*s2rx*R1*H24 + s2e*s2ry*R4*H37)

        res[2,0] = (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04)* \
                   (s1n*s1rx*Q1*H6 + s1n*s1ry*Q4*H7 - s1n*s1rz*Q9*H15) + \
                   (-s1rx*s1ry*Q9*H01 - s1rx*s1rz*Q4*H03 - s1ry*s1rz*Q1*H04)* \
                   (s1n*s1rx*Q3*H6 + s1n*s1ry*Q2*H7 - s1n*s1rz*Q5*H15) + \
                   (s1rx*s1ry*(2 - s1n)*Q5*H14 - s1rx*s1rz*(2 - s1n)*Q2*H12 - s1ry*s1rz*(2 - s1n)*Q3*H13)* \
                   (-s1rx*Q1*H02 - s1ry*Q4*H9 - s1rz*Q9*SA1 - s1xc + s2rx*R1*H10 + s2ry*R4*H11 + s2rz*R9*SA2 + s2xc) + \
                   (-s1rx*s1ry*(2 - s1n)*Q9*H14 + s1rx*s1rz*(2 - s1n)*Q4*H12 + s1ry*s1rz*(2 - s1n)*Q1*H13)* \
                   (-s1rx*Q3*H02 - s1ry*Q2*H9 - s1rz*Q5*SA1 - s1yc + s2rx*R3*H10 + s2ry*R2*H11 + s2rz*R5*SA2 + s2yc)

        res[2,1] = (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04)* \
                   (s1e*s1rx*Q1*H20 - s1e*s1ry*Q4*H35) + \
                   (-s1rx*s1ry*Q9*H01 - s1rx*s1rz*Q4*H03 - s1ry*s1rz*Q1*H04)* \
                   (s1e*s1rx*Q3*H20 - s1e*s1ry*Q2*H35) + \
                   (-s1rx*s1rz*(2 - s1e)*Q4*H8 + s1ry*s1rz*(2 - s1e)*Q1*H16)* \
                   (-s1rx*Q3*H02 - s1ry*Q2*H9 - s1rz*Q5*SA1 - s1yc + s2rx*R3*H10 + s2ry*R2*H11 + s2rz*R5*SA2 + s2yc) + \
                   (s1rx*s1rz*(2 - s1e)*Q2*H8 - s1ry*s1rz*(2 - s1e)*Q3*H16)* \
                   (-s1rx*Q1*H02 - s1ry*Q4*H9 - s1rz*Q9*SA1 - s1xc + s2rx*R1*H10 + s2ry*R4*H32 + s2rz*R9*SA2 + s2xc)

        res[2,2] = (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04)* \
                   (-s2n*s2rx*R1*H36 - s2n*s2ry*R4*H22 + s2n*s2rz*R9*H26) + \
                   (-s1rx*s1ry*Q9*H01 - s1rx*s1rz*Q4*H03 - s1ry*s1rz*Q1*H04)* \
                   (-s2n*s2rx*R3*H36 - s2n*s2ry*R2*H22 + s2n*s2rz*R5*H26)

        res[2,3] = (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04)* \
                   (-s2e*s2rx*R1*H24 + s2e*s2ry*R4*H37) + \
                   (-s1rx*s1ry*Q9*H01 - s1rx*s1rz*Q4*H03 - s1ry*s1rz*Q1*H04)* \
                   (-s2e*s2rx*R3*H24 + s2e*s2ry*R2*H37)

        res[3,0] = (
          (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04)* \
          (2*s1rx*s1ry*(2 - s1n)*Q5*H14 - 2*s1rx*s1rz*(2 - s1n)*Q2*H12 - 2*s1ry*s1rz*(2 - s1n)*Q3*H13)/2 + \
          (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04)* \
          (2*s1rx*s1ry*(2 - s1n)*Q9*H14 - 2*s1rx*s1rz*(2 - s1n)*Q4*H12 - 2*s1ry*s1rz*(2 - s1n)*Q1*H13)/2 + \
          (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04)* \
          (2*s1rx*s1ry*(2 - s1n)*Q8*H14 - 2*s1rx*s1rz*(2 - s1n)*Q7*H12 - 2*s1ry*s1rz*(2 - s1n)*Q6*H13)/2 \
        )*np.sqrt( \
          (s2rx*s2ry*R5*H18 + s2rx*s2rz*R2*H17 + s2ry*s2rz*R3*H33)**2 + \
          (s2rx*s2ry*R9*H18 + s2rx*s2rz*R4*H17 + s2ry*s2rz*R1*H33)**2 + \
          (s2rx*s2ry*R8*H18 + s2rx*s2rz*R7*H17 + s2ry*s2rz*R6*H33)**2 \
        )/np.sqrt( \
          (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04)**2 + \
          (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04)**2 + \
          (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04)**2 \
        ) + \
        (s2rx*s2ry*R5*H18 + s2rx*s2rz*R2*H17 + s2ry*s2rz*R3*H33)* \
        (s1rx*s1ry*(2 - s1n)*Q5*H14 - s1rx*s1rz*(2 - s1n)*Q2*H12 - s1ry*s1rz*(2 - s1n)*Q3*H13) + \
        (s2rx*s2ry*R9*H18 + s2rx*s2rz*R4*H17 + s2ry*s2rz*R1*H33)* \
        (s1rx*s1ry*(2 - s1n)*Q9*H14 - s1rx*s1rz*(2 - s1n)*Q4*H12 - s1ry*s1rz*(2 - s1n)*Q1*H13) + \
        (s2rx*s2ry*R8*H18 + s2rx*s2rz*R7*H17 + s2ry*s2rz*R6*H33)* \
        (s1rx*s1ry*(2 - s1n)*Q8*H14 - s1rx*s1rz*(2 - s1n)*Q7*H12 - s1ry*s1rz*(2 - s1n)*Q6*H13)

        res[3,1] = (
          (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04)* \
          (2*s1rx*s1rz*(2 - s1e)*Q2*H8 - 2*s1ry*s1rz*(2 - s1e)*Q3*H16)/2 + \
          (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04)* \
          (2*s1rx*s1rz*(2 - s1e)*Q4*H8  - 2*s1ry*s1rz*(2 - s1e)*Q1*H16)/2 + \
          (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04)* \
          (2*s1rx*s1rz*(2 - s1e)*Q7*H8 - 2*s1ry*s1rz*(2 - s1e)*Q6*H16)/2 \
        )*np.sqrt( \
          (s2rx*s2ry*R5*H18 + s2rx*s2rz*R2*H17 + s2ry*s2rz*R3*H33)**2 + \
          (s2rx*s2ry*R9*H18 + s2rx*s2rz*R4*H17 + s2ry*s2rz*R1*H33)**2 + \
          (s2rx*s2ry*R8*H18 + s2rx*s2rz*R7*H17 + s2ry*s2rz*R6*H33)**2 \
        )/np.sqrt( \
          (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04)**2 + \
          (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04)**2 + \
          (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04)**2 \
        ) + \
        (s2rx*s2ry*R5*H18 + s2rx*s2rz*R2*H17 + s2ry*s2rz*R3*H33)* \
        (s1rx*s1rz*(2 - s1e)*Q2*H8 - s1ry*s1rz*(2 - s1e)*Q3*H16) + \
        (s2rx*s2ry*R9*H18 + s2rx*s2rz*R4*H17 + s2ry*s2rz*R1*H33)* \
        (s1rx*s1rz*(2 - s1e)*Q4*H8 - s1ry*s1rz*(2 - s1e)*Q1*H16) + \
        (s2rx*s2ry*R8*H18 + s2rx*s2rz*R7*H17 + s2ry*s2rz*R6*H33)*\
        (s1rx*s1rz*(2 - s1e)*Q7*H8 - s1ry*s1rz*(2 - s1e)*Q6*H16)

        res[3,2] = (
          (s2rx*s2ry*R5*H18 + s2rx*s2rz*R2*H17 + s2ry*s2rz*R3*H33)* \
          (2*s2rx*s2ry*(2 - s2n)*R5*H27 - 2*s2rx*s2rz*(2 - s2n)*R2*H29 - 2*s2ry*s2rz*(2 - s2n)*R3*H31)/2 + \
          (s2rx*s2ry*R9*H18 + s2rx*s2rz*R4*H17 + s2ry*s2rz*R1*H33)* \
          (2*s2rx*s2ry*(2 - s2n)*R9*H27 - 2*s2rx*s2rz*(2 - s2n)*R4*H29 - 2*s2ry*s2rz*(2 - s2n)*R1*H31)/2 + \
          (s2rx*s2ry*R8*H18 + s2rx*s2rz*R7*H17 + s2ry*s2rz*R6*H33)* \
          (2*s2rx*s2ry*(2 - s2n)*R8*H27 - 2*s2rx*s2rz*(2 - s2n)*R7*H29 - 2*s2ry*s2rz*(2 - s2n)*R6*H31)/2 \
        )*np.sqrt( \
          (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04)**2 + \
          (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04)**2 + \
          (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04)**2 \
        )/np.sqrt( \
          (s2rx*s2ry*R5*H18 + s2rx*s2rz*R2*H17 + s2ry*s2rz*R3*H33)**2 + \
          (s2rx*s2ry*R9*H18 + s2rx*s2rz*R4*H17 + s2ry*s2rz*R1*H33)**2 + \
          (s2rx*s2ry*R8*H18 + s2rx*s2rz*R7*H17 + s2ry*s2rz*R6*H33)**2 \
        ) + \
        (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04)* \
        (s2rx*s2ry*(2 - s2n)*R5*H27 - s2rx*s2rz*(2 - s2n)*R2*H29 - s2ry*s2rz*(2 - s2n)*R3*H31) + \
        (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04)* \
        (s2rx*s2ry*(2 - s2n)*R9*H27 - s2rx*s2rz*(2 - s2n)*R4*H29 - s2ry*s2rz*(2 - s2n)*R1*H31) + \
        (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04)* \
        (s2rx*s2ry*(2 - s2n)*R8*H27 - s2rx*s2rz*(2 - s2n)*R7*H29 - s2ry*s2rz*(2 - s2n)*R6*H31)

        res[3,3] = (
          (s2rx*s2ry*R5*H18 + s2rx*s2rz*R2*H17 + s2ry*s2rz*R3*H33)* \
          (2*s2rx*s2rz*(2 - s2e)*R2*H34 - 2*s2ry*s2rz*(2 - s2e)*R3*H30)/2 + \
          (s2rx*s2ry*R9*H18 + s2rx*s2rz*R4*H17 + s2ry*s2rz*R1*H33)* \
          (2*s2rx*s2rz*(2 - s2e)*R4*H34 - 2*s2ry*s2rz*(2 - s2e)*R1*H30)/2 + \
          (s2rx*s2ry*R8*H18 + s2rx*s2rz*R7*H17 + s2ry*s2rz*R6*H33)* \
          (2*s2rx*s2rz*(2 - s2e)*R7*H34 - 2*s2ry*s2rz*(2 - s2e)*R6*H30)/2 \
        )*np.sqrt( \
          (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04)**2 + \
          (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04)**2 + \
          (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04)**2 \
        )/np.sqrt( \
          (s2rx*s2ry*R5*H18 + s2rx*s2rz*R2*H17 + s2ry*s2rz*R3*H33)**2 + \
          (s2rx*s2ry*R9*H18 + s2rx*s2rz*R4*H17 + s2ry*s2rz*R1*H33)**2 + \
          (s2rx*s2ry*R8*H18 + s2rx*s2rz*R7*H17 + s2ry*s2rz*R6*H33)**2 \
        ) + \
        (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04)* \
        (s2rx*s2rz*(2 - s2e)*R2*H34 - s2ry*s2rz*(2 - s2e)*R3*H30) + \
        (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04)* \
        (s2rx*s2rz*(2 - s2e)*R4*H34 - s2ry*s2rz*(2 - s2e)*R1*H30) + \
        (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04)* \
        (s2rx*s2rz*(2 - s2e)*R7*H34 - s2ry*s2rz*(2 - s2e)*R6*H30)




        ## res_ref est directement obtenu a partir du calcul formel
        ## permet de tester que la reecriture des formules n'a pas fait ajouter des erreurs
        res_ref = np.zeros((4,4))

        res_ref[0,0] = (-s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1n*s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + s1n*s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - s1n*s1rz*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) + 2*s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + 2*s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - 2*s1rz*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1n*s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + s1n*s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - s1n*s1rz*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) + 2*s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + 2*s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - 2*s1rz*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (-s1rx*s1ry*(2 - s1n)*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) + 2*s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) - 2*s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2 - s1n)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) + s1ry*s1rz*(2 - s1n)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)))*(-s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1zc + s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2zc) + (s1rx*s1ry*(2 - s1n)*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 2*s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 2*s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(2 - s1n)*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - s1ry*s1rz*(2 - s1n)*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)))*(-s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1yc + s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2yc)

        res_ref[0,1] = (-s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1e*s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - s1e*s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) + 2*s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - 2*s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1e*s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - s1e*s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) + 2*s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - 2*s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (s1rx*s1rz*(2 - s1e)*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) - s1ry*s1rz*(2 - s1e)*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))*(-s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1yc + s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2yc) + (-s1rx*s1rz*(2 - s1e)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) - 2*s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) + s1ry*s1rz*(2 - s1e)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))*(-s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1zc + s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2zc)

        res_ref[0,2] = (-s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2n*s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - s2n*s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) + s2n*s2rz*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.cos(a2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) - 2*s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - 2*s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + 2*s2rz*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.cos(a2)*np.abs(np.sin(a2))**s2n*DiracDelta(np.sin(a2))) + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2n*s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - s2n*s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) + s2n*s2rz*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) - 2*s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - 2*s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + 2*s2rz*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**s2n*DiracDelta(np.sin(a2)))

        res_ref[0,3] = (-s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2e*s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) + s2e*s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) - 2*s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + 2*s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.sin(b2))*np.sign(np.cos(a2))) + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2e*s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) + s2e*s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) - 2*s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + 2*s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.sin(b2))*np.sign(np.cos(a2)))

        res_ref[1,0] = (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1n*s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + s1n*s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - s1n*s1rz*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) + 2*s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + 2*s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - 2*s1rz*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (-s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1n*s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + s1n*s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - s1n*s1rz*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) + 2*s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + 2*s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - 2*s1rz*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (s1rx*s1ry*(2 - s1n)*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 2*s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 2*s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(2 - s1n)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - s1ry*s1rz*(2 - s1n)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)))*(-s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1zc + s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2zc) + (-s1rx*s1ry*(2 - s1n)*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) + 2*s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) - 2*s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2 - s1n)*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) + s1ry*s1rz*(2 - s1n)*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)))*(-s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1xc + s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2xc)

        res_ref[1,1] = (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1e*s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - s1e*s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) + 2*s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - 2*s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (-s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1e*s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - s1e*s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) + 2*s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - 2*s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (-s1rx*s1rz*(2 - s1e)*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) - 2*s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) + s1ry*s1rz*(2 - s1e)*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))*(-s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1xc + s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2xc) + (s1rx*s1rz*(2 - s1e)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) - s1ry*s1rz*(2 - s1e)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))*(-s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1zc + s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2zc)

        res_ref[1,2] = (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2n*s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - s2n*s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) + s2n*s2rz*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.cos(a2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) - 2*s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - 2*s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + 2*s2rz*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.cos(a2)*np.abs(np.sin(a2))**s2n*DiracDelta(np.sin(a2))) + (-s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2n*s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - s2n*s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) + s2n*s2rz*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) - 2*s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - 2*s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + 2*s2rz*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**s2n*DiracDelta(np.sin(a2)))

        res_ref[1,3] = (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2e*s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) + s2e*s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) - 2*s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + 2*s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.sin(b2))*np.sign(np.cos(a2))) + (-s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2e*s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) + s2e*s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) - 2*s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + 2*s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.sin(b2))*np.sign(np.cos(a2)))

        res_ref[2,0] = (s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1n*s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + s1n*s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - s1n*s1rz*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) + 2*s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + 2*s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - 2*s1rz*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (-s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1n*s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + s1n*s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - s1n*s1rz*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) + 2*s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + 2*s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - 2*s1rz*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (s1rx*s1ry*(2 - s1n)*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 2*s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 2*s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(2 - s1n)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - s1ry*s1rz*(2 - s1n)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)))*(-s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1xc + s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2xc) + (-s1rx*s1ry*(2 - s1n)*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) + 2*s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) - 2*s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2 - s1n)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) + s1ry*s1rz*(2 - s1n)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)))*(-s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1yc + s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2yc)

        res_ref[2,1] = (s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1e*s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - s1e*s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) + 2*s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - 2*s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (-s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1e*s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - s1e*s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) + 2*s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - 2*s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (-s1rx*s1rz*(2 - s1e)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) - 2*s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) + s1ry*s1rz*(2 - s1e)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))*(-s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1yc + s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2yc) + (s1rx*s1rz*(2 - s1e)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) - s1ry*s1rz*(2 - s1e)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))*(-s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1xc + s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2xc)

        res_ref[2,2] = (s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2n*s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - s2n*s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) + s2n*s2rz*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) - 2*s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - 2*s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + 2*s2rz*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**s2n*DiracDelta(np.sin(a2))) + (-s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2n*s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - s2n*s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) + s2n*s2rz*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) - 2*s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - 2*s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + 2*s2rz*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**s2n*DiracDelta(np.sin(a2)))

        res_ref[2,3] = (s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2e*s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) + s2e*s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) - 2*s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + 2*s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.sin(b2))*np.sign(np.cos(a2))) + (-s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2e*s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) + s2e*s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) - 2*s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + 2*s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.sin(b2))*np.sign(np.cos(a2)))

        res_ref[3,0] = ((s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(2*s1rx*s1ry*(2 - s1n)*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 4*s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 4*s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) - 2*s1rx*s1rz*(2 - s1n)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - 2*s1ry*s1rz*(2 - s1n)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)))/2 + (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(2*s1rx*s1ry*(2 - s1n)*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 4*s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 4*s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) - 2*s1rx*s1rz*(2 - s1n)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - 2*s1ry*s1rz*(2 - s1n)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)))/2 + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(2*s1rx*s1ry*(2 - s1n)*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 4*s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 4*s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) - 2*s1rx*s1rz*(2 - s1n)*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - 2*s1ry*s1rz*(2 - s1n)*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)))/2)*np.sqrt((s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2)/np.sqrt((s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2) + (s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(s1rx*s1ry*(2 - s1n)*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 2*s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 2*s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(2 - s1n)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - s1ry*s1rz*(2 - s1n)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1))) + (s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(s1rx*s1ry*(2 - s1n)*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 2*s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 2*s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(2 - s1n)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - s1ry*s1rz*(2 - s1n)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1))) + (s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(s1rx*s1ry*(2 - s1n)*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 2*s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 2*s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(2 - s1n)*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - s1ry*s1rz*(2 - s1n)*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)))

        res_ref[3,1] = ((s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(2*s1rx*s1rz*(2 - s1e)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 4*s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) - 2*s1ry*s1rz*(2 - s1e)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 4*s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))/2 + (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(2*s1rx*s1rz*(2 - s1e)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 4*s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) - 2*s1ry*s1rz*(2 - s1e)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 4*s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))/2 + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(2*s1rx*s1rz*(2 - s1e)*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 4*s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) - 2*s1ry*s1rz*(2 - s1e)*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 4*s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))/2)*np.sqrt((s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2)/np.sqrt((s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2) + (s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(s1rx*s1rz*(2 - s1e)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) - s1ry*s1rz*(2 - s1e)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1))) + (s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(s1rx*s1rz*(2 - s1e)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) - s1ry*s1rz*(2 - s1e)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1))) + (s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(s1rx*s1rz*(2 - s1e)*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) - s1ry*s1rz*(2 - s1e)*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))

        res_ref[3,2] = ((s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(2*s2rx*s2ry*(2 - s2n)*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))**2*np.sign(np.cos(a2))/np.abs(np.sin(a2)) - 4*s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.sin(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.cos(a2))*np.sign(np.sin(a2)) + 4*s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.sin(a2))*np.sign(np.cos(a2)) - 2*s2rx*s2rz*(2 - s2n)*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2))/np.abs(np.cos(a2)) - 2*s2ry*s2rz*(2 - s2n)*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2))/np.abs(np.cos(a2)))/2 + (s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(2*s2rx*s2ry*(2 - s2n)*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))**2*np.sign(np.cos(a2))/np.abs(np.sin(a2)) - 4*s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.cos(a2))*np.sign(np.sin(a2)) + 4*s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.sin(a2))*np.sign(np.cos(a2)) - 2*s2rx*s2rz*(2 - s2n)*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2))/np.abs(np.cos(a2)) - 2*s2ry*s2rz*(2 - s2n)*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2))/np.abs(np.cos(a2)))/2 + (s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(2*s2rx*s2ry*(2 - s2n)*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))**2*np.sign(np.cos(a2))/np.abs(np.sin(a2)) - 4*s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.sin(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.cos(a2))*np.sign(np.sin(a2)) + 4*s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.sin(a2))*np.sign(np.cos(a2)) - 2*s2rx*s2rz*(2 - s2n)*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2))/np.abs(np.cos(a2)) - 2*s2ry*s2rz*(2 - s2n)*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2))/np.abs(np.cos(a2)))/2)*np.sqrt((s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2)/np.sqrt((s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2) + (s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s2rx*s2ry*(2 - s2n)*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))**2*np.sign(np.cos(a2))/np.abs(np.sin(a2)) - 2*s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.sin(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.cos(a2))*np.sign(np.sin(a2)) + 2*s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.sin(a2))*np.sign(np.cos(a2)) - s2rx*s2rz*(2 - s2n)*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2))/np.abs(np.cos(a2)) - s2ry*s2rz*(2 - s2n)*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2))/np.abs(np.cos(a2))) + (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s2rx*s2ry*(2 - s2n)*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))**2*np.sign(np.cos(a2))/np.abs(np.sin(a2)) - 2*s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.cos(a2))*np.sign(np.sin(a2)) + 2*s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.sin(a2))*np.sign(np.cos(a2)) - s2rx*s2rz*(2 - s2n)*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2))/np.abs(np.cos(a2)) - s2ry*s2rz*(2 - s2n)*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2))/np.abs(np.cos(a2))) + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s2rx*s2ry*(2 - s2n)*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))**2*np.sign(np.cos(a2))/np.abs(np.sin(a2)) - 2*s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.sin(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.cos(a2))*np.sign(np.sin(a2)) + 2*s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.sin(a2))*np.sign(np.cos(a2)) - s2rx*s2rz*(2 - s2n)*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2))/np.abs(np.cos(a2)) - s2ry*s2rz*(2 - s2n)*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2))/np.abs(np.cos(a2)))

        res_ref[3,3] = ((s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(2*s2rx*s2rz*(2 - s2e)*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) + 4*s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.sin(b2)) - 2*s2ry*s2rz*(2 - s2e)*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 4*s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2)))/2 + (s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(2*s2rx*s2rz*(2 - s2e)*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) + 4*s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.sin(b2)) - 2*s2ry*s2rz*(2 - s2e)*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 4*s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2)))/2 + (s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(2*s2rx*s2rz*(2 - s2e)*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) + 4*s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.sin(b2)) - 2*s2ry*s2rz*(2 - s2e)*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 4*s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2)))/2)*np.sqrt((s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2)/np.sqrt((s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2) + (s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s2rx*s2rz*(2 - s2e)*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) + 2*s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.sin(b2)) - s2ry*s2rz*(2 - s2e)*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 2*s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2))) + (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s2rx*s2rz*(2 - s2e)*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) + 2*s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.sin(b2)) - s2ry*s2rz*(2 - s2e)*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 2*s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2))) + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s2rx*s2rz*(2 - s2e)*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) + 2*s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.sin(b2)) - s2ry*s2rz*(2 - s2e)*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 2*s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2)))
        """
        res_ref[0,0] = (-s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1n*s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + s1n*s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - s1n*s1rz*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) - 2*s1rz*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1n*s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + s1n*s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - s1n*s1rz*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) + 2*s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + 2*s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - 2*s1rz*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (-s1rx*s1ry*(2 - s1n)*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) + 2*s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) - 2*s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2 - s1n)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) + s1ry*s1rz*(2 - s1n)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)))*(-s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1zc + s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2zc) + (s1rx*s1ry*(2 - s1n)*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 2*s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 2*s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(2 - s1n)*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - s1ry*s1rz*(2 - s1n)*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)))*(-s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1yc + s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2yc)
        res_ref[0,1] = (-s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1e*s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**(s1e-1)*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2 - s1e*s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1))) + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1e*s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**(s1e-1)*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2 - s1e*s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) + 2*s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - 2*s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (s1rx*s1rz*(2 - s1e)*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) - s1ry*s1rz*(2 - s1e)*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(1 - s1e)*np.sign(np.cos(b1))**2)*(-s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1yc + s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2yc) + (-s1rx*s1rz*(2 - s1e)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) - 2*s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) + s1ry*s1rz*(2 - s1e)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(1 - s1e)*np.sign(np.cos(b1))**2 + 2*s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))*(-s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1zc + s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2zc)
        res_ref[0,2] = (-s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2n*s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - s2n*s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) + s2n*s2rz*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.cos(a2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) + 2*s2rz*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.cos(a2)*np.abs(np.sin(a2))**s2n*DiracDelta(np.sin(a2))) + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2n*s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - s2n*s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) + s2n*s2rz*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) - 2*s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - 2*s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + 2*s2rz*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**s2n*DiracDelta(np.sin(a2)))
        res_ref[0,3] = (-s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2e*s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) + s2e*s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2))) + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2e*s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) + s2e*s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) - 2*s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + 2*s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.sin(b2))*np.sign(np.cos(a2)))
        res_ref[1,0] = (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1n*s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + s1n*s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - s1n*s1rz*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) - 2*s1rz*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (-s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1n*s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + s1n*s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - s1n*s1rz*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) + 2*s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + 2*s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1))) + (s1rx*s1ry*(2 - s1n)*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - s1rx*s1rz*(2 - s1n)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - s1ry*s1rz*(2 - s1n)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)))*(-s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1zc + s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2zc) + (-s1rx*s1ry*(2 - s1n)*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) + 2*s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) - 2*s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2 - s1n)*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) + s1ry*s1rz*(2 - s1n)*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)))*(-s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1xc + s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2xc)
        res_ref[1,1] = (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1e*s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - s1e*s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1))) + (-s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1e*s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - s1e*s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) + 2*s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - 2*s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (-s1rx*s1rz*(2 - s1e)*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + s1ry*s1rz*(2 - s1e)*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)))*(-s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1xc + s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2xc) + (s1rx*s1rz*(2 - s1e)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) - s1ry*s1rz*(2 - s1e)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))*(-s1rx*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1zc + s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2zc)
        res_ref[1,2] = (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2n*s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - s2n*s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) + s2n*s2rz*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.cos(a2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) + 2*s2rz*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.cos(a2)*np.abs(np.sin(a2))**s2n*DiracDelta(np.sin(a2))) + (-s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2n*s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - s2n*s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) + s2n*s2rz*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) - 2*s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - 2*s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + 2*s2rz*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**s2n*DiracDelta(np.sin(a2)))
        res_ref[1,3] = (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2e*s2rx*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) + s2e*s2ry*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2))) + (-s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2e*s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) + s2e*s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) - 2*s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + 2*s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.sin(b2))*np.sign(np.cos(a2)))
        res_ref[2,0] = (s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1n*s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + s1n*s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - s1n*s1rz*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) + 2*s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + 2*s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1))) + (-s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1n*s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + s1n*s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - s1n*s1rz*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) + 2*s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + 2*s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - 2*s1rz*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (s1rx*s1ry*(2 - s1n)*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 2*s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 2*s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(2 - s1n)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - s1ry*s1rz*(2 - s1n)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)))*(-s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1xc + s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2xc) + (-s1rx*s1ry*(2 - s1n)*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) + s1rx*s1rz*(2 - s1n)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) + s1ry*s1rz*(2 - s1n)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)))*(-s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1yc + s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2yc)
        res_ref[2,1] = (s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1e*s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - s1e*s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) + 2*s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - 2*s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (-s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s1e*s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - s1e*s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) + 2*s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - 2*s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (-s1rx*s1rz*(2 - s1e)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) - 2*s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) + s1ry*s1rz*(2 - s1e)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))*(-s1rx*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1yc + s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2yc) + (s1rx*s1rz*(2 - s1e)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) - s1ry*s1rz*(2 - s1e)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))*(-s1rx*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - s1ry*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - s1rz*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) - s1xc + s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + s2rz*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) + s2xc)
        res_ref[2,2] = (s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2n*s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - s2n*s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) + s2n*s2rz*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) - 2*s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - 2*s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + 2*s2rz*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**s2n*DiracDelta(np.sin(a2))) + (-s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2n*s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - s2n*s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) + s2n*s2rz*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) - 2*s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - 2*s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + 2*s2rz*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**s2n*DiracDelta(np.sin(a2)))
        res_ref[2,3] = (s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2e*s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) + s2e*s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) - 2*s2rx*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + 2*s2ry*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.sin(b2))*np.sign(np.cos(a2))) + (-s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) - s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(-s2e*s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) + s2e*s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) - 2*s2rx*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + 2*s2ry*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.sin(b2))*np.sign(np.cos(a2)))
        res_ref[3,0] = ((s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(2*s1rx*s1ry*(2 - s1n)*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 4*s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 4*s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) - 2*s1rx*s1rz*(2 - s1n)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - 2*s1ry*s1rz*(2 - s1n)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)))/2 + (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(2*s1rx*s1ry*(2 - s1n)*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 4*s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 4*s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) - 2*s1rx*s1rz*(2 - s1n)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - 2*s1ry*s1rz*(2 - s1n)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)))/2 + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(2*s1rx*s1ry*(2 - s1n)*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 4*s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 4*s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) - 2*s1rx*s1rz*(2 - s1n)*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - 2*s1ry*s1rz*(2 - s1n)*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)))/2)*np.sqrt((s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2)/np.sqrt((s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2) + (s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(s1rx*s1ry*(2 - s1n)*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 2*s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 2*s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(2 - s1n)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - s1ry*s1rz*(2 - s1n)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1))) + (s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(s1rx*s1ry*(2 - s1n)*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - s1rx*s1rz*(2 - s1n)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - s1ry*s1rz*(2 - s1n)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1))) + (s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(s1rx*s1ry*(2 - s1n)*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2*np.sign(np.cos(a1))/np.abs(np.sin(a1)) - 2*s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.sin(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(a1)) + 2*s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))*np.sign(np.cos(a1)) - s1rx*s1rz*(2 - s1n)*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))/np.abs(np.cos(a1)) - s1ry*s1rz*(2 - s1n)*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))/np.abs(np.cos(a1)))
        res_ref[3,1] = ((s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(2*s1rx*s1rz*(2 - s1e)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 4*s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) - 2*s1ry*s1rz*(2 - s1e)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 4*s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))/2 + (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(2*s1rx*s1rz*(2 - s1e)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 4*s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) - 2*s1ry*s1rz*(2 - s1e)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 4*s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))/2 + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(2*s1rx*s1rz*(2 - s1e)*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 4*s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) - 2*s1ry*s1rz*(2 - s1e)*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 4*s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1)))/2)*np.sqrt((s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2)/np.sqrt((s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2) + (s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(s1rx*s1rz*(2 - s1e)*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) - s1ry*s1rz*(2 - s1e)*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1))) + (s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(s1rx*s1rz*(2 - s1e)*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) + 2*s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1)) - s1ry*s1rz*(2 - s1e)*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1))) + (s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(s1rx*s1rz*(2 - s1e)*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2/np.abs(np.sin(b1)) - s1ry*s1rz*(2 - s1e)*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)))
        res_ref[3,2] = ((s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(2*s2rx*s2ry*(2 - s2n)*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))**2*np.sign(np.cos(a2))/np.abs(np.sin(a2)) - 4*s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.sin(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.cos(a2))*np.sign(np.sin(a2)) + 4*s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.sin(a2))*np.sign(np.cos(a2)) - 2*s2rx*s2rz*(2 - s2n)*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2))/np.abs(np.cos(a2)) - 2*s2ry*s2rz*(2 - s2n)*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2))/np.abs(np.cos(a2)))/2 + (s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(2*s2rx*s2ry*(2 - s2n)*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))**2*np.sign(np.cos(a2))/np.abs(np.sin(a2)) - 4*s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.cos(a2))*np.sign(np.sin(a2)) + 4*s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.sin(a2))*np.sign(np.cos(a2)) - 2*s2rx*s2rz*(2 - s2n)*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2))/np.abs(np.cos(a2)) - 2*s2ry*s2rz*(2 - s2n)*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2))/np.abs(np.cos(a2)))/2 + (s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(2*s2rx*s2ry*(2 - s2n)*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))**2*np.sign(np.cos(a2))/np.abs(np.sin(a2)) - 4*s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.sin(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.cos(a2))*np.sign(np.sin(a2)) + 4*s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.sin(a2))*np.sign(np.cos(a2)) - 2*s2rx*s2rz*(2 - s2n)*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2))/np.abs(np.cos(a2)) - 2*s2ry*s2rz*(2 - s2n)*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2))/np.abs(np.cos(a2)))/2)*np.sqrt((s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2)/np.sqrt((s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2) + (s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s2rx*s2ry*(2 - s2n)*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))**2*np.sign(np.cos(a2))/np.abs(np.sin(a2)) - 2*s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.sin(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.cos(a2))*np.sign(np.sin(a2)) + 2*s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.sin(a2))*np.sign(np.cos(a2)) - s2rx*s2rz*(2 - s2n)*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2))/np.abs(np.cos(a2)) - s2ry*s2rz*(2 - s2n)*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2))/np.abs(np.cos(a2))) + (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s2rx*s2ry*(2 - s2n)*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))**2*np.sign(np.cos(a2))/np.abs(np.sin(a2)) - 2*s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.cos(a2))*np.sign(np.sin(a2)) + 2*s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.sin(a2))*np.sign(np.cos(a2)) - s2rx*s2rz*(2 - s2n)*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2))/np.abs(np.cos(a2)) - s2ry*s2rz*(2 - s2n)*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2))/np.abs(np.cos(a2))) + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s2rx*s2ry*(2 - s2n)*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))**2*np.sign(np.cos(a2))/np.abs(np.sin(a2)) - 2*s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.sin(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.cos(a2))*np.sign(np.sin(a2)) + 2*s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.sin(a2))*np.sign(np.cos(a2)) - s2rx*s2rz*(2 - s2n)*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2))/np.abs(np.cos(a2)) - s2ry*s2rz*(2 - s2n)*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2))/np.abs(np.cos(a2)))
        res_ref[3,3] = ((s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(2*s2rx*s2rz*(2 - s2e)*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) + 4*s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.sin(b2)) - 2*s2ry*s2rz*(2 - s2e)*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 4*s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2)))/2 + (s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(2*s2rx*s2rz*(2 - s2e)*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) + 4*s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.sin(b2)) - 2*s2ry*s2rz*(2 - s2e)*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 4*s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2)))/2 + (s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))*(2*s2rx*s2rz*(2 - s2e)*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) + 4*s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.sin(b2)) - 2*s2ry*s2rz*(2 - s2e)*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 4*s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2)))/2)*np.sqrt((s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2 + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))**2)/np.sqrt((s2rx*s2ry*(-2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*s2ry*(2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2 + (s2rx*s2ry*(s2q0**2 - s2q1**2 - s2q2**2 + s2q3**2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))*np.sign(np.cos(a2)) + s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2)) + s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2)))**2) + (s1rx*s1ry*(-2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(s1q0**2 - s1q1**2 + s1q2**2 - s1q3**2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s2rx*s2rz*(2 - s2e)*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) + 2*s2rx*s2rz*(s2q0**2 - s2q1**2 + s2q2**2 - s2q3**2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.sin(b2)) - s2ry*s2rz*(2 - s2e)*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 2*s2ry*s2rz*(2*s2q0*s2q3 + 2*s2q1*s2q2)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2))) + (s1rx*s1ry*(2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(-2*s1q0*s1q3 + 2*s1q1*s1q2)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(s1q0**2 + s1q1**2 - s1q2**2 - s1q3**2)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s2rx*s2rz*(2 - s2e)*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) + 2*s2rx*s2rz*(-2*s2q0*s2q3 + 2*s2q1*s2q2)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.sin(b2)) - s2ry*s2rz*(2 - s2e)*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 2*s2ry*s2rz*(s2q0**2 + s2q1**2 - s2q2**2 - s2q3**2)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2))) + (s1rx*s1ry*(s1q0**2 - s1q1**2 - s1q2**2 + s1q3**2)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))*np.sign(np.cos(a1)) + s1rx*s1rz*(2*s1q0*s1q1 + 2*s1q2*s1q3)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1)) + s1ry*s1rz*(-2*s1q0*s1q2 + 2*s1q1*s1q3)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(b1)))*(s2rx*s2rz*(2 - s2e)*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))**2/np.abs(np.sin(b2)) + 2*s2rx*s2rz*(2*s2q0*s2q1 + 2*s2q2*s2q3)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.sin(b2)) - s2ry*s2rz*(2 - s2e)*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 2*s2ry*s2rz*(-2*s2q0*s2q2 + 2*s2q1*s2q3)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2)))
        """
        print("grad F : res = ",res," res_ref = ",res_ref," ecart = ",np.linalg.norm(res-res_ref))

        return res

    # u0 = np.array([2*np.pi/3, 2*np.pi/3, np.pi/3, np.pi/3])
    # r = f_contacts(u0,s1,s2)
    # print("r=",r)
    # dr = grad_f_contacts(u0,s1,s2)
    # print("dr=",dr)

    def verifier_gradient(f,df,x,s1,s2):
        N = len(x)
        gg = np.zeros((N,N))
        for i in range(N): # x0, x1, ...
            eps = 1e-6
            e = np.zeros(N)
            e[i] = eps
            gg[:,i] = (f(x+e,s1,s2) - f(x-e,s1,s2))/(2*eps)
        print('erreur numerique dans le calcul du gradient: %g (doit etre petit)' % np.linalg.norm(df(x,s1,s2)-gg))
    verifier_gradient(f_contacts,grad_f_contacts,np.array([2*np.pi/3, 2*np.pi/3, np.pi/3, np.pi/3]),s1,s2)


    ## Determination de la donne initiale de la methode de Newton
    # 1/ on calcule les positions sur qq points repartis sur les deux superellipsoid2D,
    # 2/ on calcule toutes les distances
    # 3/ on cherche la distance minimale => on a la donnee initiale
    num = 10
    aa = np.linspace(-np.pi/2,np.pi/2,num=num)
    bb = np.linspace(-np.pi,np.pi,num=num)
    pts_ext_s1 = s1.surface_pt(aa,bb)
    pts_ext_s2 = s2.surface_pt(aa,bb)
    distances = np.zeros((bb.shape[0],bb.shape[0]))
    for i in range(bb.shape[0]):
        for j in range(bb.shape[0]):
            distances[i,j] = np.linalg.norm(pts_ext_s1[i,:]-pts_ext_s2[j,:])
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
    def pas_armijo(u,d,f,gradf,s1,s2):
        t = 1
        m = np.dot(d,gradf(u,s1,s2))
        alpha=0.3 # les paramètres alpha et beta comportent une valeur par défaut
        beta=0.5
        while np.linalg.norm(f(u+t*d,s1,s2)) > np.linalg.norm(f(u,s1,s2) + alpha*t*m):
            t = beta*t
        return t
    def backtrack(f,x,d,m,s1,s2,alpha=0.3,beta=0.5):
        t = 1
        while np.linalg.norm(f(x+t*d,s1,s2)) > np.linalg.norm(f(x,s1,s2) + alpha*t*m):
            t = beta*t
        return t
    def linesearch(f,x,d,s1,s2):
        t = 1
        while (np.linalg.norm(f(x+t*d,s1,s2)) > np.linalg.norm(f(x,s1,s2)))and (t>0.1) :
            t -= 0.01
        return t
    cc = 0
    itermax = 2000
    u = u0.copy()
    dk = np.ones(u.shape)

    ### test pour comparaison avec le C++
    print("s1 : xc=",s1.xc," yc=",s1.yc," zc=",s1.zc," rx=",s1.rx," ry=",s1.ry," rz=",s1.rz," e=",s1.e," n=",s1.n," q=",s1.q)
    print("s2 : xc=",s2.xc," yc=",s2.yc," zc=",s2.zc," rx=",s2.rx," ry=",s2.ry," rz=",s2.rz," e=",s2.e," n=",s2.n," q=",s2.q)
    print("u0 = ",u0)
    val1 = np.array([0.43])
    val2 = np.array([0.65])
    # print("s1.surface_pt(val1,val2)      = ",s1.surface_pt(val1,val2),     " s2.surface_pt(val2,val1)      = ",s2.surface_pt(val2,val1))
    # print("s1.surface_normal(val1,val2)  = ",s1.surface_normal(val1,val2), " s2.surface_normal(val2,val1)  = ",s2.surface_normal(val2,val1))
    # print("s1.surface_tangent(val1,val2) = ",s1.surface_tangent(val1,val2)," s2.surface_tangent(val2,val1) = ",s2.surface_tangent(val2,val1))
    # print("u=",u)
    # print("grad_f_contacts(u,s1,s2) = ",grad_f_contacts(u,s1,s2), " f_contacts(u,s1,s2) = ",f_contacts(u,s1,s2))

    while (cc<itermax) and (np.linalg.norm(dk)>1.0e-7) and (np.linalg.norm(f_contacts(u,s1,s2))>1e-10) :
        ## dk = -(gradFk)^-1 Fk : direction de descente
        dk = np.linalg.solve(grad_f_contacts(u,s1,s2), -f_contacts(u,s1,s2))
        # print("grad Fk = ",grad_f_contacts(u,s1,s2)," -Fk = ",-f_contacts(u,s1,s2))
        ## tk : pas d'armijo
        # tk = pas_armijo(u,dk,f_contacts,grad_f_contacts,s1,s2)
        # tk = backtrack(f_contacts,u,dk,-np.linalg.norm(dk)**2,s1,s2)
        tk = linesearch(f_contacts,u,dk,s1,s2)
        u += tk*dk
        print("iteration ",cc," dk = ",dk," tk = ",tk," u = ",u," |dk| = ",np.linalg.norm(dk)," np.cost=",np.linalg.norm(f_contacts(u,s1,s2)))
        cc += 1
    b_final = u

    final_pt1 = s1.surface_pt(np.array([b_final[0]]),np.array([b_final[1]]))
    normal_final_pt1 = s1.surface_normal(np.array([b_final[0]]),np.array([b_final[1]]))
    final_pt2 = s2.surface_pt(np.array([b_final[2]]),np.array([b_final[3]]))
    normal_final_pt2 = s2.surface_normal(np.array([b_final[2]]),np.array([b_final[3]]))

    print("final_pt1 = ",final_pt1)
    print("final_pt2 = ",final_pt2)
    sys.exit()

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

    """
    add_mesh(mesh, color=None, style=None, scalars=None, clim=None, show_edges=None,
    edge_color=None, point_size=5.0, line_width=None, opacity=1.0, flip_scalars=False,
    lighting=None, n_colors=256, interpolate_before_map=True, cmap=None, label=None,
    reset_camera=None, scalar_bar_args=None, show_scalar_bar=None, stitle=None,
    multi_colors=False, name=None, texture=None, render_points_as_spheres=None,
    render_lines_as_tubes=False, smooth_shading=None, ambient=0.0, diffuse=1.0,
    specular=0.0, specular_power=100.0, nan_color=None, nan_opacity=1.0, culling=None,
    rgb=False, categories=False, use_transparency=False, below_color=None,
    above_color=None, annotations=None, pickable=True, preference='point',
    log_scale=False, render=True, **kwargs)

    pyvista.plot(var_item, off_screen=None, full_screen=False, screenshot=None,
    interactive=True, cpos=None, window_size=None, show_bounds=False, show_axes=True,
    notebook=None, background=None, text='', return_img=False, eye_dome_lighting=False,
    volume=False, parallel_projection=False, use_ipyvtk=None, **kwargs)
    """
