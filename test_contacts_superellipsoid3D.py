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
        # nx = np.array( self.ry*self.rz*np.abs(np.cos(a))**(2-self.n)*np.sign(np.cos(b))*np.abs(np.cos(b))**(2-self.e) )
        # ny = np.array( self.rx*self.rz*np.abs(np.cos(a))**(2-self.n)*np.sign(np.sin(b))*np.abs(np.sin(b))**(2-self.e) )
        # nz = np.array( self.rx*self.ry*np.sign(np.cos(a))*np.sign(np.sin(a))*np.abs(np.sin(a))**(2-self.n) )

        nx = np.array( self.ry*self.rz*np.sign(np.cos(a))*np.abs(np.cos(a))**(2-self.n)*np.sign(np.cos(b))*np.abs(np.cos(b))**(2-self.e) )
        ny = np.array( self.rx*self.rz*np.sign(np.cos(a))*np.abs(np.cos(a))**(2-self.n)*np.sign(np.sin(b))*np.abs(np.sin(b))**(2-self.e) )
        nz = np.array( self.rx*self.ry*np.sign(np.sin(a))*np.abs(np.sin(a))**(2-self.n) )

        # nx = np.array( np.sign(np.cos(a))*self.ry*self.rz*np.cos(a)**2*np.cos(b) )
        # ny = np.array( np.sign(np.cos(a))*self.rx*self.rz*np.cos(a)**2*np.sin(b) )
        # nz = np.array( np.sign(np.cos(a))*self.rx*self.ry*np.cos(a)*np.sin(a) )

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
        # s1q0,s1q1,s1q2,s1q3 = sympy.symbols("s1q0,s1q1,s1q2,s1q3",real=True)
        # s1Mr = sympy.Matrix([[ s1q0**2+s1q1**2-s1q2**2-s1q3**2, 2*s1q1*s1q2-2*s1q0*s1q3, 2*s1q1*s1q3+2*s1q0*s1q2, 0 ],
        #                    [ 2*s1q1*s1q2+2*s1q0*s1q3, s1q0**2-s1q1**2+s1q2**2-s1q3**2, 2*s1q2*s1q3-2*s1q0*s1q1, 0 ],
        #                    [ 2*s1q1*s1q3-2*s1q0*s1q2, 2*s1q2*s1q3+2*s1q0*s1q1, s1q0**2-s1q1**2-s1q2**2+s1q3**2, 0 ],
        #                    [ 0,0,0,1]])
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
        ## v1 : pb de signe
        # s1nx = s1ry*s1rz*sympy.Abs(cos(a1))**(2-s1n)*sign(cos(b1))*sympy.Abs(cos(b1))**(2-s1e)
        # s1ny = s1rx*s1rz*sympy.Abs(cos(a1))**(2-s1n)*sign(sin(b1))*sympy.Abs(sin(b1))**(2-s1e)
        # s1nz = s1rx*s1ry*sign(cos(a1))*sign(sin(a1))*sympy.Abs(sin(a1))**(2-s1n)
        # v2 : apres multiplication par sign(cos(a))
        s1nx = s1ry*s1rz*sign(cos(a1))*sympy.Abs(cos(a1))**(2-s1n)*sign(cos(b1))*sympy.Abs(cos(b1))**(2-s1e)
        s1ny = s1rx*s1rz*sign(cos(a1))*sympy.Abs(cos(a1))**(2-s1n)*sign(sin(b1))*sympy.Abs(sin(b1))**(2-s1e)
        s1nz = s1rx*s1ry*sign(sin(a1))*sympy.Abs(sin(a1))**(2-s1n)
        n1 = s1Mr@sympy.Matrix([s1nx,s1ny,s1nz,1])
        print("\ncalcul formel : pt1 = \n",pt1)
        print("\ncalcul formel : n1 = \n",n1)
        # superellipsoid2
        s2e = sympy.symbols("s2e",real=True)
        s2n = sympy.symbols("s2n",real=True)
        # s2q0,s2q1,s2q2,s2q3 = sympy.symbols("s2q0,s2q1,s2q2,s2q3",real=True)
        # s2Mr = sympy.Matrix([[ s2q0**2+s2q1**2-s2q2**2-s2q3**2, 2*s2q1*s2q2-2*s2q0*s2q3, 2*s2q1*s2q3+2*s2q0*s2q2, 0 ],
        #                    [ 2*s2q1*s2q2+2*s2q0*s2q3, s2q0**2-s2q1**2+s2q2**2-s2q3**2, 2*s2q2*s2q3-2*s2q0*s2q1, 0 ],
        #                    [ 2*s2q1*s2q3-2*s2q0*s2q2, 2*s2q2*s2q3+2*s2q0*s2q1, s2q0**2-s2q1**2-s2q2**2+s2q3**2, 0 ],
        #                    [ 0,0,0,1]])
        N00, N01, N02, N10, N11, N12, N20, N21, N22 = sympy.symbols("N00, N01, N02, N10, N11, N12, N20, N21, N22",real=True)
        s2Mr = sympy.Matrix([[ N00, N01, N02, 0 ],
                             [ N10, N11, N12, 0 ],
                             [ N20, N21, N22, 0 ],
                             [ 0,0,0,1]])
        s2rx,s2ry,s2rz,s2xc,s2yc,s2zc = sympy.symbols("s2rx,s2ry,s2rz,s2xc,s2yc,s2zc",real=True)
        s2Ms= sympy.Matrix([[s2rx,0,0,0],[0,s2ry,0,0],[0,0,s2rz,0],[0,0,0,1]])
        s2Mt = sympy.Matrix([[1,0,0,s2xc],[0,1,0,s2yc],[0,0,1,s2zc],[0,0,0,1]])
        s2M = s2Mt@s2Mr@s2Ms
        s2x = sign(cos(a2))*sympy.Abs(cos(a2))**s2n*sign(cos(b2))*sympy.Abs(cos(b2))**s2e
        s2y = sign(cos(a2))*sympy.Abs(cos(a2))**s2n*sign(sin(b2))*sympy.Abs(sin(b2))**s2e
        s2z = sign(sin(a2))*sympy.Abs(sin(a2))**s2n
        pt2 = s2M@sympy.Matrix([s2x,s2y,s2z,1])
        ## v1 : pb de signe
        # s2nx = s2ry*s2rz*sympy.Abs(cos(a2))**(2-s2n)*sign(cos(b2))*sympy.Abs(cos(b2))**(2-s2e)
        # s2ny = s2rx*s2rz*sympy.Abs(cos(a2))**(2-s2n)*sign(sin(b2))*sympy.Abs(sin(b2))**(2-s2e)
        # s2nz = s2rx*s2ry*sign(cos(a2))*sign(sin(a2))*sympy.Abs(sin(a2))**(2-s2n)
        # v2 : apres multiplication par sign(cos(a))
        s2nx = s2ry*s2rz*sign(cos(a2))*sympy.Abs(cos(a2))**(2-s2n)*sign(cos(b2))*sympy.Abs(cos(b2))**(2-s2e)
        s2ny = s2rx*s2rz*sign(cos(a2))*sympy.Abs(cos(a2))**(2-s2n)*sign(sin(b2))*sympy.Abs(sin(b2))**(2-s2e)
        s2nz = s2rx*s2ry*sign(sin(a2))*sympy.Abs(sin(a2))**(2-s2n)
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

    def create_binit_xy(n, s, liste_b=None):
        if(liste_b == None):
            liste_b = []
        angles = np.linspace(0, 0.5*np.pi, num=n, endpoint=True)
        for b_ell in angles:
            x_ell =  s.rx*np.cos(b_ell)
            y_ell =  s.ry*np.sin(b_ell)
            d = ( (s.ry*x_ell)**(2/s.e)+(s.rx*y_ell)**(2/s.e) )**(s.e/2)
            x_supell = x_ell*s.rx*s.ry/d
            y_supell = y_ell*s.rx*s.ry/d
            sinb = max(-1.0, min( 1.0, np.sqrt( ((y_supell/s.ry)**2)**(1/s.e) )) )
            b = np.arcsin(sinb)
            ## b doit varier entre -pi et pi
            if (b>np.pi):
                b -= 2*np.pi
            if (b<-np.pi):
                b += 2*np.pi
            liste_b.append(b)
            liste_b.append(b+np.pi/2)
            liste_b.append(b-np.pi)
            liste_b.append(b-np.pi/2)
        return liste_b

    def create_ainit_yz(n, s, liste_a=None):
        if(liste_a == None):
            liste_a = []
        angles = np.linspace(0, np.pi/2, num=n, endpoint=True)
        for a_ell in angles:
            y_ell =  s.ry*np.cos(a_ell)
            z_ell =  s.rz*np.sin(a_ell)
            # d = ( (s.rz*y_ell)**(2/s.e)+(s.ry*z_ell)**(2/s.e) )**(s.e/2)
            d = ( (s.rz*y_ell)**(2/s.n)+(s.ry*z_ell)**(2/s.n) )**(s.n/2)
            y_supell = y_ell*s.ry*s.rz/d
            z_supell = z_ell*s.ry*s.rz/d
            # sina = max(-1.0, min( 1.0, np.sqrt( ((z_supell/s.rz)**2)**(1/s.e) )) )
            sina = max(-1.0, min( 1.0, np.sqrt( ((z_supell/s.rz)**2)**(1/s.n) )) )
            a = np.arcsin(sina)
            liste_a.append(a)
            liste_a.append(a+np.pi/2)
            liste_a.append(a-np.pi)
            liste_a.append(a-np.pi/2)
        return liste_a

    def create_ainit_xz(n, s, liste_a=None):
        if(liste_a == None):
            liste_a = []
        angles = np.linspace(0, np.pi/2, num=n, endpoint=True)
        for a_ell in angles:
            x_ell =  s.rx*np.cos(a_ell)
            z_ell =  s.rz*np.sin(a_ell)
            # d = ( (s.rz*x_ell)**(2/s.e)+(s.rx*z_ell)**(2/s.e) )**(s.e/2)
            d = ( (s.rz*x_ell)**(2/s.n)+(s.rx*z_ell)**(2/s.n) )**(s.n/2)
            x_supell = x_ell*s.rx*s.rz/d
            z_supell = z_ell*s.rx*s.rz/d
            # sina = max(-1.0, min( 1.0, np.sqrt( ((z_supell/s.rz)**2)**(1/s.e) )) )
            sina = max(-1.0, min( 1.0, np.sqrt( ((z_supell/s.rz)**2)**(1/s.n) )) )
            a = np.arcsin(sina)
            liste_a.append(a)
            liste_a.append(a+np.pi/2)
            liste_a.append(a-np.pi)
            liste_a.append(a-np.pi/2)
        return liste_a

    def test_initialisation():
        s = SuperEllipsoid3D(1,-2,-1,   8,2,1,  n=0.1, e=1, theta=-np.pi/5, w=[2,1,1])
        # s = SuperEllipsoid3D(1,-2,-1,   8,2,1,  n=1, e=1)
        # s = SuperEllipsoid3D(1,-2,-1,   2,2,2,  n=1, e=1)
        p = pv.Plotter(window_size=[2400,1350])
        L = 10
        grid = pv.UniformGrid()
        arr = np.arange((2*L)**3).reshape((2*L,2*L,2*L))
        grid.dimensions = np.array(arr.shape) + 1 #dim + 1 because cells
        grid.origin = (-L, -L, -L)
        grid.spacing = (1, 1, 1)
        p.add_mesh(grid, show_edges=True, opacity=0.1)
        p.add_mesh(s.mesh(),color="red", opacity=0.5, name="s")

        binit = np.array( create_binit_xy(16, s) )
        print("binit = ",binit)
        pts = s.surface_pt(np.zeros_like(binit),binit)
        normals = s.surface_normal(np.zeros_like(binit),binit)
        nodes = pv.PolyData(pts)
        nodes["normal"] = normals
        p.add_mesh(nodes.glyph(factor=0.1, geom=pv.Sphere()),color="blue", name="pts")
        p.add_mesh(nodes.glyph(orient="normal",factor=0.25, geom=pv.Arrow()),color="blue", name="normals")

        ainit_yz = np.array( create_ainit_yz(16, s) )
        print("ainit_yz = ",ainit_yz)
        pts_yz = s.surface_pt(ainit_yz, np.pi/2*np.ones_like(ainit_yz))
        normals_yz = s.surface_normal(ainit_yz, np.pi/2*np.ones_like(ainit_yz))
        nodes_yz = pv.PolyData(pts_yz)
        nodes_yz["normal"] = normals_yz
        p.add_mesh(nodes_yz.glyph(factor=0.1, geom=pv.Sphere()),color="magenta", name="pts_yz")
        p.add_mesh(nodes_yz.glyph(orient="normal",factor=0.25, geom=pv.Arrow()),color="magenta", name="normals_yz")

        ainit_xz = np.array( create_ainit_xz(16, s) )
        print("ainit_xz = ",ainit_xz)
        pts_xz = s.surface_pt(ainit_xz, np.zeros_like(ainit_xz))
        normals_xz = s.surface_normal(ainit_xz, np.zeros_like(ainit_xz))
        nodes_xz = pv.PolyData(pts_xz)
        nodes_xz["normal"] = normals_xz
        p.add_mesh(nodes_xz.glyph(factor=0.1, geom=pv.Sphere()),color="brown", name="pts_xz")
        p.add_mesh(nodes_xz.glyph(orient="normal",factor=0.25, geom=pv.Arrow()),color="brown", name="normals_xz")

        ppp = []
        nnn = []
        for bb in binit:
            for ayz in ainit_yz:
                ppp.append(s.surface_pt([ayz],[bb]))
                nnn.append(s.surface_normal([ayz],[bb]))
            for axz in ainit_xz:
                ppp.append(s.surface_pt([axz],[bb]))
                nnn.append(s.surface_normal([axz],[bb]))
        ppp = np.array(ppp)
        nnn = np.array(nnn)
        nodes_all = pv.PolyData(ppp)
        nodes_all["normal"] = nnn
        p.add_mesh(nodes_all.glyph(factor=0.1, geom=pv.Sphere()),color="yellow", name="pts_all")
        p.add_mesh(nodes_all.glyph(orient="normal",factor=0.25, geom=pv.Arrow()),color="yellow", name="normals_all")

        num = 17
        theta1=np.pi/2
        theta2=0
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
        pts3 = s.surface_pt(angles,np.zeros_like(angles))
        nodes3 = pv.PolyData(pts3)
        p.show_grid()
        p.show_bounds()
        p.camera_position = 'xy'
        p.camera.SetParallelProjection(True)
        p.show(window_size=(1600,900))



    test_initialisation()
    sys.exit()

    # s1 = SuperEllipsoid3D(0,0,0, 1,1,1, n=1, e=0.5, theta=80, w=[0,0,1])
    # s2 = SuperEllipsoid3D(2,2,3, 1,3,2, n=0.6, e=1, theta=80, w = [1,1,1])

    # s1 = SuperEllipsoid3D(0,0,0, 2,1,1, n=0.9, e=0.8, theta=65, w=[0,-1,1])
    # s2 = SuperEllipsoid3D(5,4,4, 1,1,1, n=0.4, e=0.9, theta=14, w = [1,1,1])
    s1 = SuperEllipsoid3D(-0.2, 0., 0.,   .1, .05, .05, n=1, e=1, theta=np.pi/4, w=[0,0,1])
    s2 = SuperEllipsoid3D(0.2, 0., 0.,    .1, .05, .05, n=1, e=1, theta=np.pi/4, w = [0,0,1])
    # # s1 = SuperEllipsoid3D(-0.2, 0., 0.,   .1, .05, .05, n=1, e=1, theta=np.pi/4, w=[0,0,1])
    # s2 = SuperEllipsoid3D(0.2, 0., 0.,    .1, .05, .05, n=1, e=1, theta=np.pi/4, w = [0,0,1])
    # scopi::superellipsoid<dim> s1({{-0.2, 0., 0.}}, {scopi::quaternion(PI/4)}, {{.1, .05, .05}}, {{1, 1}});
    # scopi::superellipsoid<dim> s2({{0.2, 0., 0.}}, {scopi::quaternion(-PI/4)}, {{.1, .05, .05}}, {{1, 1}});


    #s1.calcul_formel()
    #sys.exit()
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
    # p.add_mesh(nodes1.glyph(factor=0.05, geom=pv.Sphere()),color="blue")
    # p.add_mesh(nodes1.glyph(orient="normal",factor=0.25, geom=pv.Arrow()),color="blue")
    # p.add_mesh(nodes1.glyph(orient="tangent1",factor=0.25, geom=pv.Arrow()),color="magenta")
    # p.add_mesh(nodes1.glyph(orient="tangent2",factor=0.25, geom=pv.Arrow()),color="magenta")

    pts2 = s2.surface_pt(aa,bb)
    normals2 = s2.surface_normal(aa,bb)
    tgt21,tgt22 = s2.surface_tangent(aa,bb)
    nodes2 = pv.PolyData(pts2)
    nodes2["normal"] = normals2
    nodes2["tangent1"] = tgt21
    nodes2["tangent2"] = tgt22
    # p.add_mesh(nodes2.glyph(factor=0.05, geom=pv.Sphere()),color="blue")
    # p.add_mesh(nodes2.glyph(orient="normal",factor=0.25, geom=pv.Arrow()),color="blue")
    # p.add_mesh(nodes2.glyph(orient="tangent1",factor=0.25, geom=pv.Arrow()),color="magenta")
    # p.add_mesh(nodes2.glyph(orient="tangent2",factor=0.25, geom=pv.Arrow()),color="magenta")

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
        # res = np.zeros((4,))
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
        M00 = (s1q00 + s1q11 - s1q22 - s1q33)  # M1[0,0]
        M01 = (-s1q03 + s1q12) # M1[0,1]
        M02 = (2*s1q0*s1q2 + 2*s1q1*s1q3)    # M1[0,2]
        M10 = (s1q03 + s1q12)  # M1[1,0]
        M11 = (s1q00 - s1q11 + s1q22 - s1q33)  # M1[1,1]
        M12 = (-2*s1q0*s1q1 + 2*s1q2*s1q3)  # M1[1,2]
        M20 = (-2*s1q0*s1q2 + 2*s1q1*s1q3)  # M1[2,0]
        M21 = (2*s1q0*s1q1 + 2*s1q2*s1q3)   # M1[2,1]
        M22 = (s1q00 - s1q11 - s1q22 + s1q33)   # M1[2,2]
        N00 = (s2q00 + s2q11 - s2q22 - s2q33)  # M2[0,0]
        N01 = (-s2q03 + s2q12) # M2[0,1]
        N02 = (2*s2q0*s2q2 + 2*s2q1*s2q3)    # M2[0,2]
        N10 = (s2q03 + s2q12)  # M2[1,0]
        N11 = (s2q00 - s2q11 + s2q22 - s2q33)  # M2[1,1]
        N12 = (-2*s2q0*s2q1 + 2*s2q2*s2q3)  # M2[1,2]
        N20 = (-2*s2q0*s2q2 + 2*s2q1*s2q3)  # M2[2,0]
        N21 = (2*s2q0*s2q1 + 2*s2q2*s2q3)   # M2[2,1]
        N22 = (s2q00 - s2q11 - s2q22 + s2q33)   # M2[2,2]

        ca1 = np.cos(a1)
        ca1n = np.sign(ca1)*np.abs(ca1)**s1n
        ca1n2 = np.sign(ca1)*np.abs(ca1)**(2 - s1n)

        sa1 = np.sin(a1)
        sa1n = np.sign(sa1)*np.abs(sa1)**s1n
        sa1n2 = np.sign(sa1)*np.abs(sa1)**(2 - s1n)

        cb1 = np.cos(b1)
        cb1e = np.sign(cb1)*np.abs(cb1)**s1e
        cb1e2 = np.sign(cb1)*np.abs(cb1)**(2 - s1e)

        sb1 = np.sin(b1)
        sb1e = np.sign(sb1)*np.abs(sb1)**s1e
        sb1e2 = np.sign(sb1)*np.abs(sb1)**(2 - s1e)

        A1 = s1ry*s1rz*ca1n2*cb1e2
        A2 = s1rx*s1rz*sb1e2*ca1n2
        A3 = s1rx*s1ry*sa1n2
        A4 = s1rx*ca1n*cb1e
        A5 = s1ry*sb1e*ca1n
        A6 = s1rz*sa1n

        ca2 = np.cos(a2)
        ca2n = np.sign(ca2)*np.abs(ca2)**s2n
        ca2n2 = np.sign(ca2)*np.abs(ca2)**(2 - s2n)

        sa2 = np.sin(a2)
        sa2n = np.sign(sa2)*np.abs(sa2)**s2n
        sa2n2 = np.sign(sa2)*np.abs(sa2)**(2 - s2n)

        cb2 = np.cos(b2)
        cb2e = np.sign(cb2)*np.abs(cb2)**s2e
        cb2e2 = np.sign(cb2)*np.abs(cb2)**(2 - s2e)

        sb2 = np.sin(b2)
        sb2e = np.sign(sb2)*np.abs(sb2)**s2e
        sb2e2 = np.sign(sb2)*np.abs(sb2)**(2 - s2e)

        B1 = s2rx*ca2n*cb2e
        B2 = s2ry*sb2e*ca2n
        B3 = s2rz*sa2n
        B4 = s2ry*s2rz*ca2n2*cb2e2
        B5 = s2rx*s2rz*sb2e2*ca2n2
        B6 = s2rx*s2ry*sa2n2

        # B1 = s2rx * ca2n *cb2e
        # B2 = s2ry * sb2e *ca2n
        # B3 = s2rz * sa2n
        # B4 = s2ry * s2rz * ca2n2 * cb2e2
        # B5 = s2rx * s2rz * sb2e2 * ca2n2
        # B6 = s2rx * s2ry * sa2n2 * np.sign(ca2)

        res = np.zeros((4,))

        res[0] = - (M10*A1 + M11*A2 + M12*A3)*(-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc) \
                 + (M20*A1 + M21*A2 + M22*A3)*(-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc)

        res[1] =   (M00*A1 + M01*A2 + M02*A3)*(-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc) \
                 - (M20*A1 + M21*A2 + M22*A3)*(-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc)

        res[2] = - (M00*A1 + M01*A2 + M02*A3)*(-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc) \
                 + (M10*A1 + M11*A2 + M12*A3)*(-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc)

        res[3] =  (M00*A1 + M01*A2 + M02*A3)*(N00*B4 + N01*B5 + N02*B6) \
                + (M10*A1 + M11*A2 + M12*A3)*(N10*B4 + N11*B5 + N12*B6) \
                + (M20*A1 + M21*A2 + M22*A3)*(N20*B4 + N21*B5 + N22*B6) \
                + np.sqrt((M00*A1 + M01*A2 + M02*A3)**2 + (M10*A1 + M11*A2 + M12*A3)**2 + (M20*A1 + M21*A2 + M22*A3)**2) \
                * np.sqrt((N00*B4 + N01*B5 + N02*B6)**2 + (N10*B4 + N11*B5 + N12*B6)**2 + (N20*B4 + N21*B5 + N22*B6)**2)

        ## res_ref est directement obtenu a partir du calcul formel
        ## permet de tester que la reecriture des formules n'a pas fait ajouter des erreurs
        res_ref = np.zeros((4,))

        res_ref[0] = -(M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-M20*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M21*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M22*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N20*s2rx*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N21*s2ry*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N22*s2rz*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) - s1zc + s2zc) + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-M10*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M11*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M12*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N10*s2rx*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N11*s2ry*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N12*s2rz*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) - s1yc + s2yc)

        res_ref[1] = (M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-M20*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M21*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M22*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N20*s2rx*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N21*s2ry*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N22*s2rz*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) - s1zc + s2zc) - (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-M00*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M01*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M02*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N00*s2rx*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N01*s2ry*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N02*s2rz*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) - s1xc + s2xc)

        res_ref[2] = -(M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-M10*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M11*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M12*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N10*s2rx*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N11*s2ry*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N12*s2rz*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) - s1yc + s2yc) + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-M00*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M01*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M02*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N00*s2rx*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N01*s2ry*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N02*s2rz*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) - s1xc + s2xc)

        res_ref[3] =  (M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(N00*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N01*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N02*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))) + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(N10*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N11*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N12*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))) + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(N20*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N21*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N22*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))) + np.sqrt((M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))**2 + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))**2 + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))**2)*np.sqrt((N00*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N01*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N02*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))**2 + (N10*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N11*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N12*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))**2 + (N20*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N21*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N22*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))**2)

        print("F : res = ",res," res_ref = ",res_ref," ecart = ",np.linalg.norm(res-res_ref))

        return res

    def DiracDelta(x):
        return 0

    def grad_f_contacts(u,s1,s2):
        # res = np.zeros((4,4))
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
        M00 = (s1q00 + s1q11 - s1q22 - s1q33)  # M1[0,0]
        M01 = (-s1q03 + s1q12) # M1[0,1]
        M02 = (2*s1q0*s1q2 + 2*s1q1*s1q3)    # M1[0,2]
        M10 = (s1q03 + s1q12)  # M1[1,0]
        M11 = (s1q00 - s1q11 + s1q22 - s1q33)  # M1[1,1]
        M12 = (-2*s1q0*s1q1 + 2*s1q2*s1q3)  # M1[1,2]
        M20 = (-2*s1q0*s1q2 + 2*s1q1*s1q3)  # M1[2,0]
        M21 = (2*s1q0*s1q1 + 2*s1q2*s1q3)   # M1[2,1]
        M22 = (s1q00 - s1q11 - s1q22 + s1q33)   # M1[2,2]
        N00 = (s2q00 + s2q11 - s2q22 - s2q33)  # M2[0,0]
        N01 = (-s2q03 + s2q12) # M2[0,1]
        N02 = (2*s2q0*s2q2 + 2*s2q1*s2q3)    # M2[0,2]
        N10 = (s2q03 + s2q12)  # M2[1,0]
        N11 = (s2q00 - s2q11 + s2q22 - s2q33)  # M2[1,1]
        N12 = (-2*s2q0*s2q1 + 2*s2q2*s2q3)  # M2[1,2]
        N20 = (-2*s2q0*s2q2 + 2*s2q1*s2q3)  # M2[2,0]
        N21 = (2*s2q0*s2q1 + 2*s2q2*s2q3)   # M2[2,1]
        N22 = (s2q00 - s2q11 - s2q22 + s2q33)   # M2[2,2]


        ca1 = np.cos(a1)
        ca1n = np.sign(ca1)*np.abs(ca1)**s1n
        ca1n1 = s1n*np.abs(ca1)**(s1n - 1)
        ca1n2 = np.sign(ca1)*np.abs(ca1)**(2 - s1n)
        ca1n3 = (2 - s1n)*np.abs(ca1)**(1 - s1n)

        sa1 = np.sin(a1)
        sa1n = np.sign(sa1)*np.abs(sa1)**s1n
        sa1n1 = s1n*np.abs(sa1)**(s1n - 1)
        sa1n2 = np.sign(sa1)*np.abs(sa1)**(2 - s1n)
        sa1n3 = (2 - s1n)*np.abs(sa1)**(1 - s1n)

        cb1 = np.cos(b1)
        cb1e = np.sign(cb1)*np.abs(cb1)**s1e
        cb1e1 = s1e*np.abs(cb1)**(s1e - 1)
        cb1e2 = np.sign(cb1)*np.abs(cb1)**(2 - s1e)
        cb1e3 = (2 - s1e)*np.abs(cb1)**(1 - s1e)

        sb1 = np.sin(b1)
        sb1e = np.sign(sb1)*np.abs(sb1)**s1e
        sb1e1 = s1e*np.abs(sb1)**(s1e - 1)
        sb1e2 = np.sign(sb1)*np.abs(sb1)**(2 - s1e)
        sb1e3 = (2 - s1e)*np.abs(sb1)**(1 - s1e)

        A1 = s1ry*s1rz*ca1n2*cb1e2
        A2 = s1rx*s1rz*sb1e2*ca1n2
        A3 = s1rx*s1ry*sa1n2
        A4 = s1rx*ca1n*cb1e
        A5 = s1ry*sb1e*ca1n
        A6 = s1rz*sa1n
        A7 = s1rx*sa1*ca1n1*cb1e
        A8 = s1ry*sa1*sb1e*ca1n1
        A9 = s1rz*ca1*sa1n1
        A10 = s1ry*s1rz*sa1*ca1n3*cb1e2
        A11 = s1rx*s1rz*sa1*sb1e2*ca1n3
        A12 = s1rx*s1ry*ca1*sa1n3
        A13 = s1rx*sb1*ca1n*cb1e1
        A14 = s1ry*cb1*sb1e1*ca1n
        A15 = s1ry*s1rz*sb1*ca1n2*cb1e3
        A16 = s1rx*s1rz*cb1*sb1e3*ca1n2

        ca2 = np.cos(a2)
        ca2n = np.sign(ca2)*np.abs(ca2)**s2n
        ca2n1 = s2n*np.abs(ca2)**(s2n - 1)
        ca2n2 = np.sign(ca2)*np.abs(ca2)**(2 - s2n)
        ca2n3 = (2 - s2n)*np.abs(ca2)**(1 - s2n)

        sa2 = np.sin(a2)
        sa2n = np.sign(sa2)*np.abs(sa2)**s2n
        sa2n1 = s2n*np.abs(sa2)**(s2n - 1)
        sa2n2 = np.sign(sa2)*np.abs(sa2)**(2 - s2n)
        sa2n3 = (2 - s2n)*np.abs(sa2)**(1 - s2n)

        cb2 = np.cos(b2)
        cb2e = np.sign(cb2)*np.abs(cb2)**s2e
        cb2e1 = s2e*np.abs(cb2)**(s2e - 1)
        cb2e2 = np.sign(cb2)*np.abs(cb2)**(2 - s2e)
        cb2e3 = (2 - s2e)*np.abs(cb2)**(1 - s2e)

        sb2 = np.sin(b2)
        sb2e = np.sign(sb2)*np.abs(sb2)**s2e
        sb2e1 = s2e*np.abs(sb2)**(s2e - 1)
        sb2e2 = np.sign(sb2)*np.abs(sb2)**(2 - s2e)
        sb2e3 = (2 - s2e)*np.abs(sb2)**(1 - s2e)

        B1 = s2rx*ca2n*cb2e
        B2 = s2ry*sb2e*ca2n
        B3 = s2rz*sa2n
        B4 = s2ry*s2rz*ca2n2*cb2e2
        B5 = s2rx*s2rz*sb2e2*ca2n2
        B6 = s2rx*s2ry*sa2n2
        B7 = s2rx*sa2*ca2n1*cb2e
        B8 = s2ry*sa2*sb2e*ca2n1
        B9 = s2rz*ca2*sa2n1
        B10 = s2rx*sb2*ca2n*cb2e1
        B11 = s2ry*cb2*sb2e1*ca2n
        B12 = s2ry*s2rz*sa2*ca2n3*cb2e2
        B13 = s2rx*s2rz*sa2*sb2e2*ca2n3
        B14 = s2rx*s2ry*ca2*sa2n3
        B15 = s2ry*s2rz*sb2*ca2n2*cb2e3
        B16 = s2rx*s2rz*cb2*sb2e3*ca2n2

        res= np.zeros((4,4))

        # res[0,0] =   (-M10*A1 - M11*A2 - M12*A3) * (M20*A7 + M21*A8 - M22*A9) \
        #            + ( M20*A1 + M21*A2 + M22*A3) * (M10*A7 + M11*A8 - M12*A9) \
        #            + ( M10*A10 + M11*A11 - M12*A12) * (-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc) \
        #            + (-M20*A10 - M21*A11 + M22*A12) * (-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc)
        #
        # res[0,1] =   (-M10*A1 - M11*A2 - M12*A3) * (M20*A13 - M21*A14) \
        #            + ( M20*A1 + M21*A2 + M22*A3) * (M10*A13 - M11*A14) \
        #            + ( M10*A15 - M11*A16) * (-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc) \
        #            + (-M20*A15 + M21*A16)*(-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc)
        #
        # res[0,2] =   (-M10*A1 - M11*A2 - M12*A3) * (-N20*B7 - N21*B8 + N22*B9) \
        #            + ( M20*A1 + M21*A2 + M22*A3) * (-N10*B7 - N11*B8 + N12*B9)
        #
        # res[0,3] =   (-M10*A1 - M11*A2 - M12*A3) * (-N20*B10 + N21*B11) \
        #            + ( M20*A1 + M21*A2 + M22*A3) * (-N10*B10 + N11*B11)
        #
        # res[1,0] =   ( M00*A1 + M01*A2 + M02*A3) * (M20*A7 + M21*A8 - M22*A9) \
        #            + (-M20*A1 - M21*A2 - M22*A3) * (M00*A7 + M01*A8 - M02*A9) \
        #            + (-M00*A10 - M01*A11 + M02*A12) * (-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc) \
        #            + ( M20*A10 + M21*A11 - M22*A12) * (-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc)
        #
        # res[1,1] =   ( M00*A1 + M01*A2 + M02*A3) * (M20*A13 - M21*A14) \
        #            + (-M20*A1 - M21*A2 - M22*A3) * (M00*A13 - M01*A14) \
        #            + (-M00*A15 + M01*A16) * (-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc) \
        #            + ( M20*A15 - M21*A16) * (-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc)
        #
        # res[1,2] =   ( M00*A1 + M01*A2 + M02*A3) * (-N20*B7 - N21*B8 + N22*B9) \
        #            + (-M20*A1 - M21*A2 - M22*A3) * (-N00*B7 - N01*B8 + N02*B9)
        #
        # res[1,3] =   ( M00*A1 + M01*A2 + M02*A3) * (-N20*B10 + N21*B11) \
        #            + (-M20*A1 - M21*A2 - M22*A3) * (-N00*B10 + N01*B11)
        #
        # res[2,0] =   (-M00*A1 - M01*A2 - M02*A3) * (M10*A7 + M11*A8 - M12*A9) \
        #            + ( M10*A1 + M11*A2 + M12*A3) * (M00*A7 + M01*A8 - M02*A9) \
        #            + ( M00*A10 + M01*A11 - M02*A12) * (-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc) \
        #            + (-M10*A10 - M11*A11 + M12*A12) * (-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc)
        #
        # res[2,1] =   (-M00*A1 - M01*A2 - M02*A3) * (M10*A13 - M11*A14) \
        #            + ( M10*A1 + M11*A2 + M12*A3) * (M00*A13 - M01*A14) \
        #            + ( M00*A15 - M01*A16) * (-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc) \
        #            + (-M10*A15 + M11*A16) * (-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc)
        #
        # res[2,2] =   (-M00*A1 - M01*A2 - M02*A3) * (-N10*B7 - N11*B8 + N12*B9) \
        #            + (M10*A1 + M11*A2 + M12*A3) * (-N00*B7 - N01*B8 + N02*B9)
        #
        # res[2,3] =   (-M00*A1 - M01*A2 - M02*A3) * (-N10*B10 + N11*B11) \
        #            + (M10*A1 + M11*A2 + M12*A3) * (-N00*B10 + N01*B11)
        #
        # res[3,0] = (   ( M00*A1 + M01*A2 + M02*A3) * (-M00*A10 - M01*A11 + M02*A12) \
        #              + ( M10*A1 + M11*A2 + M12*A3) * (-M10*A10 - M11*A11 + M12*A12) \
        #              + ( M20*A1 + M21*A2 + M22*A3) * (-M20*A10 - M21*A11 + M22*A12) \
        #            ) \
        #            * np.sqrt( (N00*B4 + N01*B5 + N02*B6)**2 + (N10*B4 + N11*B5 + N12*B6)**2 + (N20*B4 + N21*B5 + N22*B6)**2 )\
        #            / np.sqrt( (M00*A1 + M01*A2 + M02*A3)**2 + (M10*A1 + M11*A2 + M12*A3)**2 + (M20*A1 + M21*A2 + M22*A3)**2) \
        #            + ( N00*B4 + N01*B5 + N02*B6) * ( -M00*A10 - M01*A11 + M02*A12) \
        #            + ( N10*B4 + N11*B5 + N12*B6) * (-M10*A10 - M11*A11 + M12*A12) \
        #            + ( N20*B4 + N21*B5 + N22*B6) * (-M20*A10 - M21*A11 + M22*A12)
        #
        # res[3,1] = (   ( M00*A1 + M01*A2 + M02*A3) * (-M00*A15 + M01*A16) \
        #              + ( M10*A1 + M11*A2 + M12*A3) * (-M10*A15 + M11*A16) \
        #              + ( M20*A1 + M21*A2 + M22*A3) * (-M20*A15 + M21*A16) \
        #            ) \
        #            * np.sqrt( (N00*B4 + N01*B5 + N02*B6)**2 + (N10*B4 + N11*B5 + N12*B6)**2 + (N20*B4 + N21*B5 + N22*B6)**2 ) \
        #            / np.sqrt( (M00*A1 + M01*A2 + M02*A3)**2 + (M10*A1 + M11*A2 + M12*A3)**2 + (M20*A1 + M21*A2 + M22*A3)**2 ) \
        #            + ( N00*B4 + N01*B5 + N02*B6) * (-M00*A15 + M01*A16 ) \
        #            + ( N10*B4 + N11*B5 + N12*B6) * (-M10*A15 + M11*A16) \
        #            + ( N20*B4 + N21*B5 + N22*B6) * (-M20*A15 + M21*A16)
        #
        # res[3,2] =  (   ( N00*B4 + N01*B5 + N02*B6) * (-N00*B12 - N01*B13 + N02*B14) \
        #               + ( N10*B4 + N11*B5 + N12*B6) * (-N10*B12 - N11*B13 + N12*B14) \
        #               + ( N20*B4 + N21*B5 + N22*B6) * (-N20*B12 - N21*B13 + N22*B14) \
        #             ) \
        #             * np.sqrt( (M00*A1 + M01*A2 + M02*A3)**2 + (M10*A1 + M11*A2 + M12*A3)**2 + (M20*A1 + M21*A2 + M22*A3)**2 ) \
        #             / np.sqrt( (N00*B4 + N01*B5 + N02*B6)**2 + (N10*B4 + N11*B5 + N12*B6)**2 + (N20*B4 + N21*B5 + N22*B6)**2 ) \
        #             + (M00*A1 + M01*A2 + M02*A3) * (-N00*B12 - N01*B13 + N02*B14) \
        #             + (M10*A1 + M11*A2 + M12*A3) * (-N10*B12 - N11*B13 + N12*B14) \
        #             + (M20*A1 + M21*A2 + M22*A3) * (-N20*B12 - N21*B13 + N22*B14)
        #
        # res[3,3] = (    (N00*B4 + N01*B5 + N02*B6) * (-N00*B15 + N01*B16) \
        #               + (N10*B4 + N11*B5 + N12*B6) * (-N10*B15 + N11*B16) \
        #               + (N20*B4 + N21*B5 + N22*B6) * (-N20*B15 + N21*B16) \
        #            ) \
        #            * np.sqrt( (M00*A1 + M01*A2 + M02*A3)**2 + (M10*A1 + M11*A2 + M12*A3)**2 + (M20*A1 + M21*A2 + M22*A3)**2 ) \
        #            / np.sqrt( (N00*B4 + N01*B5 + N02*B6)**2 + (N10*B4 + N11*B5 + N12*B6)**2 + (N20*B4 + N21*B5 + N22*B6)**2 ) \
        #            + (M00*A1 + M01*A2 + M02*A3) * (-N00*B15 + N01*B16) \
        #            + (M10*A1 + M11*A2 + M12*A3) * (-N10*B15 + N11*B16) \
        #            + (M20*A1 + M21*A2 + M22*A3) * (-N20*B15 + N21*B16)



        res[0,0] =  (-M10*A1 - M11*A2 - M12*A3)*(M20*A7 + M21*A8 - M22*A9) \
                  + (M20*A1 + M21*A2 + M22*A3)*(M10*A7 + M11*A8 - M12*A9) \
                  + (M10*A10 + M11*A11 - M12*A12) * (-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc) \
                  + (-M20*A10 - M21*A11 + M22*A12)*(-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc)

        res[0,1] =  (-M10*A1 - M11*A2 - M12*A3)*(M20*A13 - M21*A14) \
                  + (M20*A1 + M21*A2 + M22*A3)*(M10*A13 - M11*A14) \
                  + (M10*A15 - M11*A16)*(-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc) \
                  + (-M20*A15 + M21*A16)*(-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc)

        res[0,2] =  (-M10*A1 - M11*A2 - M12*A3)*(-N20*B7 - N21*B8 + N22*B9) \
                  + (M20*A1 + M21*A2 + M22*A3)*(-N10*B7 - N11*B8 + N12*B9)

        res[0,3] =  (-M10*A1 - M11*A2 - M12*A3)*(-N20*B10 + N21*B11) \
                  + (M20*A1 + M21*A2 + M22*A3)*(-N10*B10 + N11*B11)

        res[1,0] =  (M00*A1 + M01*A2 + M02*A3)*(M20*A7 + M21*A8 - M22*A9) \
                  + (-M20*A1 - M21*A2 - M22*A3)*(M00*A7 + M01*A8 - M02*A9) \
                  + (-M00*A10 - M01*A11 + M02*A12)*(-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc) \
                  + (M20*A10 + M21*A11 - M22*A12)*(-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc)

        res[1,1] =  (M00*A1 + M01*A2 + M02*A3)*(M20*A13 - M21*A14) \
                  + (-M20*A1 - M21*A2 - M22*A3)*(M00*A13 - M01*A14) \
                  + (-M00*A15 + M01*A16)*(-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc) \
                  + (M20*A15 - M21*A16)*(-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc)

        res[1,2] =  (M00*A1 + M01*A2 + M02*A3)*(-N20*B7 - N21*B8 + N22*B9) \
                  + (-M20*A1 - M21*A2 - M22*A3)*(-N00*B7 - N01*B8 + N02*B9)

        res[1,3] =  (M00*A1 + M01*A2 + M02*A3)*(-N20*B10 + N21*B11) \
                  + (-M20*A1 - M21*A2 - M22*A3)*(-N00*B10 + N01*B11)

        res[2,0] =  (-M00*A1 - M01*A2 - M02*A3)*(M10*A7 + M11*A8 - M12*A9) \
                  + (M10*A1 + M11*A2 + M12*A3)*(M00*A7 + M01*A8 - M02*A9) \
                  + (M00*A10 + M01*A11 - M02*A12)*(-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc) \
                  + (-M10*A10 - M11*A11 + M12*A12)*(-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc)

        res[2,1] =  (-M00*A1 - M01*A2 - M02*A3)*(M10*A13 - M11*A14) \
                  + (M10*A1 + M11*A2 + M12*A3)*(M00*A13 - M01*A14) \
                  + (M00*A15 - M01*A16)*(-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc) \
                  + (-M10*A15 + M11*A16)*(-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc)

        res[2,2] =  (-M00*A1 - M01*A2 - M02*A3)*(-N10*B7 - N11*B8 + N12*B9) \
                  + (M10*A1 + M11*A2 + M12*A3)*(-N00*B7 - N01*B8 + N02*B9)

        res[2,3] =  (-M00*A1 - M01*A2 - M02*A3)*(-N10*B10 + N11*B11) \
                  + (M10*A1 + M11*A2 + M12*A3)*(-N00*B10 + N01*B11)

        res[3,0] =  ((M00*A1 + M01*A2 + M02*A3)*(-M00*A10 - M01*A11 + M02*A12) \
                  + (M10*A1 + M11*A2 + M12*A3)*(-M10*A10 - M11*A11 + M12*A12) \
                  + (M20*A1 + M21*A2 + M22*A3)*(-M20*A10 - M21*A11 + M22*A12)) \
                  * np.sqrt((N00*B4 + N01*B5 + N02*B6)**2 + (N10*B4 + N11*B5 + N12*B6)**2 + (N20*B4 + N21*B5 + N22*B6)**2) \
                  / np.sqrt((M00*A1 + M01*A2 + M02*A3)**2 + (M10*A1 + M11*A2 + M12*A3)**2 + (M20*A1 + M21*A2 + M22*A3)**2) \
                  + (N00*B4 + N01*B5 + N02*B6)*(-M00*A10 - M01*A11 + M02*A12) \
                  + (N10*B4 + N11*B5 + N12*B6)*(-M10*A10 - M11*A11 + M12*A12) \
                  + (N20*B4 + N21*B5 + N22*B6)*(-M20*A10 - M21*A11 + M22*A12)

        res[3,1] =  ((M00*A1 + M01*A2 + M02*A3)*(-M00*A15 + M01*A16) \
                   + (M10*A1 + M11*A2 + M12*A3)*(-M10*A15 + M11*A16) \
                   + (M20*A1 + M21*A2 + M22*A3)*(-M20*A15 + M21*A16)) \
                  * np.sqrt((N00*B4 + N01*B5 + N02*B6)**2 + (N10*B4 + N11*B5 + N12*B6)**2 + (N20*B4 + N21*B5 + N22*B6)**2) \
                  / np.sqrt((M00*A1 + M01*A2 + M02*A3)**2 + (M10*A1 + M11*A2 + M12*A3)**2 + (M20*A1 + M21*A2 + M22*A3)**2) \
                  + (N00*B4 + N01*B5 + N02*B6)*(-M00*A15 + M01*A16) + \
                    (N10*B4 + N11*B5 + N12*B6)*(-M10*A15 + M11*A16) + \
                    (N20*B4 + N21*B5 + N22*B6)*(-M20*A15 + M21*A16)

        res[3,2] = ((N00*B4 + N01*B5 + N02*B6)*(-N00*B12 - N01*B13 + N02*B14) \
                  + (N10*B4 + N11*B5 + N12*B6)*(-N10*B12 - N11*B13 + N12*B14) \
                  + (N20*B4 + N21*B5 + N22*B6)*(-N20*B12 - N21*B13 + N22*B14)) \
                  * np.sqrt((M00*A1 + M01*A2 + M02*A3)**2 + (M10*A1 + M11*A2 + M12*A3)**2 + (M20*A1 + M21*A2 + M22*A3)**2) \
                  / np.sqrt((N00*B4 + N01*B5 + N02*B6)**2 + (N10*B4 + N11*B5 + N12*B6)**2 + (N20*B4 + N21*B5 + N22*B6)**2) \
                  + (M00*A1 + M01*A2 + M02*A3)*(-N00*B12 - N01*B13 + N02*B14) \
                  + (M10*A1 + M11*A2 + M12*A3)*(-N10*B12 - N11*B13 + N12*B14) \
                  + (M20*A1 + M21*A2 + M22*A3)*(-N20*B12 - N21*B13 + N22*B14)

        res[3,3] =  ((N00*B4 + N01*B5 + N02*B6)*(-N00*B15 + N01*B16) \
                   + (N10*B4 + N11*B5 + N12*B6)*(-N10*B15 + N11*B16) \
                   + (N20*B4 + N21*B5 + N22*B6)*(-N20*B15 + N21*B16)) \
                   * np.sqrt((M00*A1 + M01*A2 + M02*A3)**2 \
                           + (M10*A1 + M11*A2 + M12*A3)**2 \
                           + (M20*A1 + M21*A2 + M22*A3)**2) \
                   / np.sqrt((N00*B4 + N01*B5 + N02*B6)**2 \
                           + (N10*B4 + N11*B5 + N12*B6)**2 \
                           + (N20*B4 + N21*B5 + N22*B6)**2) \
                   + (M00*A1 + M01*A2 + M02*A3)*(-N00*B15 + N01*B16) \
                   + (M10*A1 + M11*A2 + M12*A3)*(-N10*B15 + N11*B16) \
                   + (M20*A1 + M21*A2 + M22*A3)*(-N20*B15 + N21*B16)

        ## res_ref est directement obtenu a partir du calcul formel
        ## permet de tester que la reecriture des formules n'a pas fait ajouter des erreurs
        res_ref = np.zeros((4,4))

        res_ref[0,0] =  (-M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(M20*s1n*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + 2*M20*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1n*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) + 2*M21*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - M22*s1n*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) - 2*M22*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(M10*s1n*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + 2*M10*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1n*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) + 2*M11*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - M12*s1n*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) - 2*M12*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (M10*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + 2*M10*s1ry*s1rz*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) + 2*M11*s1rx*s1rz*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - M12*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) - 2*M12*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1)))*(-M20*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M21*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M22*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N20*s2rx*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N21*s2ry*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N22*s2rz*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) - s1zc + s2zc) + (-M20*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) - 2*M20*s1ry*s1rz*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) - M21*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - 2*M21*s1rx*s1rz*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) + M22*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) + 2*M22*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1)))*(-M10*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M11*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M12*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N10*s2rx*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N11*s2ry*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N12*s2rz*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) - s1yc + s2yc)

        res_ref[0,1] = (-M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(M20*s1e*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*M20*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - M21*s1e*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) - 2*M21*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(M10*s1e*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*M10*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - M11*s1e*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) - 2*M11*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (M10*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*M10*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - M11*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) - 2*M11*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1))*np.sign(np.cos(a1)))*(-M20*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M21*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M22*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N20*s2rx*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N21*s2ry*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N22*s2rz*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) - s1zc + s2zc) + (-M20*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*M20*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) + M21*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) + 2*M21*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1))*np.sign(np.cos(a1)))*(-M10*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M11*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M12*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N10*s2rx*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N11*s2ry*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N12*s2rz*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) - s1yc + s2yc)

        res_ref[0,2] = (-M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-N20*s2n*s2rx*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - 2*N20*s2rx*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - N21*s2n*s2ry*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) - 2*N21*s2ry*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + N22*s2n*s2rz*np.cos(a2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) + 2*N22*s2rz*np.cos(a2)*np.abs(np.sin(a2))**s2n*DiracDelta(np.sin(a2))) + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-N10*s2n*s2rx*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - 2*N10*s2rx*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - N11*s2n*s2ry*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) - 2*N11*s2ry*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + N12*s2n*s2rz*np.cos(a2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) + 2*N12*s2rz*np.cos(a2)*np.abs(np.sin(a2))**s2n*DiracDelta(np.sin(a2)))

        res_ref[0,3] =  (-M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-N20*s2e*s2rx*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 2*N20*s2rx*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + N21*s2e*s2ry*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) + 2*N21*s2ry*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.sin(b2))*np.sign(np.cos(a2))) + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-N10*s2e*s2rx*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 2*N10*s2rx*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + N11*s2e*s2ry*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) + 2*N11*s2ry*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.sin(b2))*np.sign(np.cos(a2)))

        res_ref[1,0] = (M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(M20*s1n*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + 2*M20*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1n*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) + 2*M21*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - M22*s1n*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) - 2*M22*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (-M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(M00*s1n*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + 2*M00*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1n*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) + 2*M01*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - M02*s1n*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) - 2*M02*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (-M00*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) - 2*M00*s1ry*s1rz*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) - M01*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - 2*M01*s1rx*s1rz*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) + M02*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) + 2*M02*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1)))*(-M20*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M21*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M22*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N20*s2rx*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N21*s2ry*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N22*s2rz*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) - s1zc + s2zc) + (M20*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + 2*M20*s1ry*s1rz*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) + 2*M21*s1rx*s1rz*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - M22*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) - 2*M22*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1)))*(-M00*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M01*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M02*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N00*s2rx*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N01*s2ry*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N02*s2rz*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) - s1xc + s2xc)

        res_ref[1,1] = (M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(M20*s1e*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*M20*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - M21*s1e*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) - 2*M21*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (-M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(M00*s1e*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*M00*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - M01*s1e*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) - 2*M01*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (-M00*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*M00*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) + M01*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) + 2*M01*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1))*np.sign(np.cos(a1)))*(-M20*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M21*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M22*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N20*s2rx*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N21*s2ry*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N22*s2rz*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) - s1zc + s2zc) + (M20*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*M20*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - M21*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) - 2*M21*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1))*np.sign(np.cos(a1)))*(-M00*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M01*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M02*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N00*s2rx*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N01*s2ry*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N02*s2rz*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) - s1xc + s2xc)

        res_ref[1,2] = (M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-N20*s2n*s2rx*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - 2*N20*s2rx*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - N21*s2n*s2ry*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) - 2*N21*s2ry*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + N22*s2n*s2rz*np.cos(a2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) + 2*N22*s2rz*np.cos(a2)*np.abs(np.sin(a2))**s2n*DiracDelta(np.sin(a2))) + (-M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-N00*s2n*s2rx*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - 2*N00*s2rx*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - N01*s2n*s2ry*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) - 2*N01*s2ry*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + N02*s2n*s2rz*np.cos(a2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) + 2*N02*s2rz*np.cos(a2)*np.abs(np.sin(a2))**s2n*DiracDelta(np.sin(a2)))

        res_ref[1,3] = (M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-N20*s2e*s2rx*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 2*N20*s2rx*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + N21*s2e*s2ry*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) + 2*N21*s2ry*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.sin(b2))*np.sign(np.cos(a2))) + (-M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-N00*s2e*s2rx*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 2*N00*s2rx*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + N01*s2e*s2ry*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) + 2*N01*s2ry*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.sin(b2))*np.sign(np.cos(a2)))

        res_ref[2,0] = (-M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(M10*s1n*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + 2*M10*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1n*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) + 2*M11*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - M12*s1n*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) - 2*M12*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(M00*s1n*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + 2*M00*s1rx*np.sin(a1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1n*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) + 2*M01*s1ry*np.sin(a1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - M02*s1n*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) - 2*M02*s1rz*np.cos(a1)*np.abs(np.sin(a1))**s1n*DiracDelta(np.sin(a1))) + (M00*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) + 2*M00*s1ry*s1rz*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) + 2*M01*s1rx*s1rz*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) - M02*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) - 2*M02*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1)))*(-M10*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M11*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M12*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N10*s2rx*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N11*s2ry*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N12*s2rz*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) - s1yc + s2yc) + (-M10*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) - 2*M10*s1ry*s1rz*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) - M11*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - 2*M11*s1rx*s1rz*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) + M12*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) + 2*M12*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1)))*(-M00*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M01*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M02*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N00*s2rx*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N01*s2ry*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N02*s2rz*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) - s1xc + s2xc)

        res_ref[2,1] = (-M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(M10*s1e*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*M10*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - M11*s1e*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) - 2*M11*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(M00*s1e*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*M00*s1rx*np.sin(b1)*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - M01*s1e*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) - 2*M01*s1ry*np.cos(b1)*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (M00*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) + 2*M00*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) - M01*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) - 2*M01*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1))*np.sign(np.cos(a1)))*(-M10*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M11*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M12*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N10*s2rx*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N11*s2ry*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N12*s2rz*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) - s1yc + s2yc) + (-M10*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*M10*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) + M11*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) + 2*M11*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1))*np.sign(np.cos(a1)))*(-M00*s1rx*np.abs(np.cos(a1))**s1n*np.abs(np.cos(b1))**s1e*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M01*s1ry*np.abs(np.sin(b1))**s1e*np.abs(np.cos(a1))**s1n*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M02*s1rz*np.abs(np.sin(a1))**s1n*np.sign(np.sin(a1)) + N00*s2rx*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N01*s2ry*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N02*s2rz*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2)) - s1xc + s2xc)

        res_ref[2,2] =  (-M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-N10*s2n*s2rx*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - 2*N10*s2rx*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - N11*s2n*s2ry*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) - 2*N11*s2ry*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + N12*s2n*s2rz*np.cos(a2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) + 2*N12*s2rz*np.cos(a2)*np.abs(np.sin(a2))**s2n*DiracDelta(np.sin(a2))) + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-N00*s2n*s2rx*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - 2*N00*s2rx*np.sin(a2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - N01*s2n*s2ry*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) - 2*N01*s2ry*np.sin(a2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + N02*s2n*s2rz*np.cos(a2)*np.abs(np.sin(a2))**s2n*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) + 2*N02*s2rz*np.cos(a2)*np.abs(np.sin(a2))**s2n*DiracDelta(np.sin(a2)))

        res_ref[2,3] = (-M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) - M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) - M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-N10*s2e*s2rx*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 2*N10*s2rx*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + N11*s2e*s2ry*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) + 2*N11*s2ry*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.sin(b2))*np.sign(np.cos(a2))) + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-N00*s2e*s2rx*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 2*N00*s2rx*np.sin(b2)*np.abs(np.cos(a2))**s2n*np.abs(np.cos(b2))**s2e*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + N01*s2e*s2ry*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) + 2*N01*s2ry*np.cos(b2)*np.abs(np.sin(b2))**s2e*np.abs(np.cos(a2))**s2n*DiracDelta(np.sin(b2))*np.sign(np.cos(a2)))

        res_ref[3,0] = ((M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-2*M00*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) - 4*M00*s1ry*s1rz*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) - 2*M01*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - 4*M01*s1rx*s1rz*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) + 2*M02*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) + 4*M02*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1)))/2 + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-2*M10*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) - 4*M10*s1ry*s1rz*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) - 2*M11*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - 4*M11*s1rx*s1rz*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) + 2*M12*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) + 4*M12*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1)))/2 + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-2*M20*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) - 4*M20*s1ry*s1rz*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) - 2*M21*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - 4*M21*s1rx*s1rz*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) + 2*M22*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) + 4*M22*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1)))/2)*np.sqrt((N00*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N01*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N02*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))**2 + (N10*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N11*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N12*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))**2 + (N20*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N21*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N22*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))**2)/np.sqrt((M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))**2 + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))**2 + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))**2) + (N00*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N01*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N02*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))*(-M00*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) - 2*M00*s1ry*s1rz*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) - M01*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - 2*M01*s1rx*s1rz*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) + M02*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) + 2*M02*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))) + (N10*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N11*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N12*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))*(-M10*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) - 2*M10*s1ry*s1rz*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) - M11*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - 2*M11*s1rx*s1rz*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) + M12*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) + 2*M12*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1))) + (N20*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N21*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N22*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))*(-M20*s1ry*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))**2*np.sign(np.cos(b1))/np.abs(np.cos(a1)) - 2*M20*s1ry*s1rz*np.sin(a1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(a1))*np.sign(np.cos(b1)) - M21*s1rx*s1rz*(2 - s1n)*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1))**2/np.abs(np.cos(a1)) - 2*M21*s1rx*s1rz*np.sin(a1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.cos(a1))*np.sign(np.sin(b1)) + M22*s1rx*s1ry*(2 - s1n)*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1))**2/np.abs(np.sin(a1)) + 2*M22*s1rx*s1ry*np.cos(a1)*np.abs(np.sin(a1))**(2 - s1n)*DiracDelta(np.sin(a1)))

        res_ref[3,1] =  ((M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-2*M00*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 4*M00*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) + 2*M01*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) + 4*M01*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1))*np.sign(np.cos(a1)))/2 + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-2*M10*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 4*M10*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) + 2*M11*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) + 4*M11*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1))*np.sign(np.cos(a1)))/2 + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-2*M20*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 4*M20*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) + 2*M21*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) + 4*M21*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1))*np.sign(np.cos(a1)))/2)*np.sqrt((N00*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N01*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N02*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))**2 + (N10*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N11*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N12*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))**2 + (N20*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N21*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N22*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))**2)/np.sqrt((M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))**2 + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))**2 + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))**2) + (N00*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N01*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N02*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))*(-M00*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*M00*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) + M01*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) + 2*M01*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (N10*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N11*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N12*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))*(-M10*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*M10*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) + M11*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) + 2*M11*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1))*np.sign(np.cos(a1))) + (N20*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N21*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N22*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))*(-M20*s1ry*s1rz*(2 - s1e)*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1))**2/np.abs(np.cos(b1)) - 2*M20*s1ry*s1rz*np.sin(b1)*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*DiracDelta(np.cos(b1))*np.sign(np.cos(a1)) + M21*s1rx*s1rz*(2 - s1e)*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))**2*np.sign(np.cos(a1))/np.abs(np.sin(b1)) + 2*M21*s1rx*s1rz*np.cos(b1)*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*DiracDelta(np.sin(b1))*np.sign(np.cos(a1)))

        res_ref[3,2] = ((N00*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N01*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N02*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))*(-2*N00*s2ry*s2rz*(2 - s2n)*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - 4*N00*s2ry*s2rz*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - 2*N01*s2rx*s2rz*(2 - s2n)*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) - 4*N01*s2rx*s2rz*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + 2*N02*s2rx*s2ry*(2 - s2n)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) + 4*N02*s2rx*s2ry*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.sin(a2)))/2 + (N10*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N11*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N12*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))*(-2*N10*s2ry*s2rz*(2 - s2n)*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - 4*N10*s2ry*s2rz*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - 2*N11*s2rx*s2rz*(2 - s2n)*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) - 4*N11*s2rx*s2rz*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + 2*N12*s2rx*s2ry*(2 - s2n)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) + 4*N12*s2rx*s2ry*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.sin(a2)))/2 + (N20*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N21*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N22*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))*(-2*N20*s2ry*s2rz*(2 - s2n)*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - 4*N20*s2ry*s2rz*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - 2*N21*s2rx*s2rz*(2 - s2n)*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) - 4*N21*s2rx*s2rz*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + 2*N22*s2rx*s2ry*(2 - s2n)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) + 4*N22*s2rx*s2ry*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.sin(a2)))/2)*np.sqrt((M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))**2 + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))**2 + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))**2)/np.sqrt((N00*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N01*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N02*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))**2 + (N10*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N11*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N12*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))**2 + (N20*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N21*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N22*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))**2) + (M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-N00*s2ry*s2rz*(2 - s2n)*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - 2*N00*s2ry*s2rz*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - N01*s2rx*s2rz*(2 - s2n)*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) - 2*N01*s2rx*s2rz*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + N02*s2rx*s2ry*(2 - s2n)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) + 2*N02*s2rx*s2ry*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.sin(a2))) + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-N10*s2ry*s2rz*(2 - s2n)*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - 2*N10*s2ry*s2rz*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - N11*s2rx*s2rz*(2 - s2n)*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) - 2*N11*s2rx*s2rz*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + N12*s2rx*s2ry*(2 - s2n)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) + 2*N12*s2rx*s2ry*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.sin(a2))) + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-N20*s2ry*s2rz*(2 - s2n)*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))**2*np.sign(np.cos(b2))/np.abs(np.cos(a2)) - 2*N20*s2ry*s2rz*np.sin(a2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(a2))*np.sign(np.cos(b2)) - N21*s2rx*s2rz*(2 - s2n)*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2))**2/np.abs(np.cos(a2)) - 2*N21*s2rx*s2rz*np.sin(a2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.cos(a2))*np.sign(np.sin(b2)) + N22*s2rx*s2ry*(2 - s2n)*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2))**2/np.abs(np.sin(a2)) + 2*N22*s2rx*s2ry*np.cos(a2)*np.abs(np.sin(a2))**(2 - s2n)*DiracDelta(np.sin(a2)))

        res_ref[3,3] =  ((N00*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N01*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N02*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))*(-2*N00*s2ry*s2rz*(2 - s2e)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 4*N00*s2ry*s2rz*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + 2*N01*s2rx*s2rz*(2 - s2e)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) + 4*N01*s2rx*s2rz*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.sin(b2))*np.sign(np.cos(a2)))/2 + (N10*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N11*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N12*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))*(-2*N10*s2ry*s2rz*(2 - s2e)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 4*N10*s2ry*s2rz*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + 2*N11*s2rx*s2rz*(2 - s2e)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) + 4*N11*s2rx*s2rz*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.sin(b2))*np.sign(np.cos(a2)))/2 + (N20*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N21*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N22*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))*(-2*N20*s2ry*s2rz*(2 - s2e)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 4*N20*s2ry*s2rz*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + 2*N21*s2rx*s2rz*(2 - s2e)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) + 4*N21*s2rx*s2rz*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.sin(b2))*np.sign(np.cos(a2)))/2)*np.sqrt((M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))**2 + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))**2 + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))**2)/np.sqrt((N00*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N01*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N02*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))**2 + (N10*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N11*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N12*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))**2 + (N20*s2ry*s2rz*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2)) + N21*s2rx*s2rz*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))*np.sign(np.cos(a2)) + N22*s2rx*s2ry*np.abs(np.sin(a2))**(2 - s2n)*np.sign(np.sin(a2)))**2) + (M00*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M01*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M02*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-N00*s2ry*s2rz*(2 - s2e)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 2*N00*s2ry*s2rz*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + N01*s2rx*s2rz*(2 - s2e)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) + 2*N01*s2rx*s2rz*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.sin(b2))*np.sign(np.cos(a2))) + (M10*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M11*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M12*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-N10*s2ry*s2rz*(2 - s2e)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 2*N10*s2ry*s2rz*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + N11*s2rx*s2rz*(2 - s2e)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) + 2*N11*s2rx*s2rz*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.sin(b2))*np.sign(np.cos(a2))) + (M20*s1ry*s1rz*np.abs(np.cos(a1))**(2 - s1n)*np.abs(np.cos(b1))**(2 - s1e)*np.sign(np.cos(a1))*np.sign(np.cos(b1)) + M21*s1rx*s1rz*np.abs(np.sin(b1))**(2 - s1e)*np.abs(np.cos(a1))**(2 - s1n)*np.sign(np.sin(b1))*np.sign(np.cos(a1)) + M22*s1rx*s1ry*np.abs(np.sin(a1))**(2 - s1n)*np.sign(np.sin(a1)))*(-N20*s2ry*s2rz*(2 - s2e)*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*np.sign(np.cos(a2))*np.sign(np.cos(b2))**2/np.abs(np.cos(b2)) - 2*N20*s2ry*s2rz*np.sin(b2)*np.abs(np.cos(a2))**(2 - s2n)*np.abs(np.cos(b2))**(2 - s2e)*DiracDelta(np.cos(b2))*np.sign(np.cos(a2)) + N21*s2rx*s2rz*(2 - s2e)*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*np.sign(np.sin(b2))**2*np.sign(np.cos(a2))/np.abs(np.sin(b2)) + 2*N21*s2rx*s2rz*np.cos(b2)*np.abs(np.sin(b2))**(2 - s2e)*np.abs(np.cos(a2))**(2 - s2n)*DiracDelta(np.sin(b2))*np.sign(np.cos(a2)))

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
    # num = 10
    # aa = np.linspace(-np.pi/2,np.pi/2,num=num)
    # bb = np.linspace(-np.pi,np.pi,num=num)
    # pts_ext_s1 = s1.surface_pt(aa,bb)
    # pts_ext_s1_normal = s1.surface_normal(aa,bb)
    # pts_ext_s2 = s2.surface_pt(aa,bb)
    # pts_ext_s2_normal = s2.surface_normal(aa,bb)
    # distances = np.zeros((bb.shape[0],bb.shape[0]))
    # for i in range(bb.shape[0]):
    #     for j in range(bb.shape[0]):
    #         distances[i,j] = np.linalg.norm(pts_ext_s1[i,:]-pts_ext_s2[j,:])
    # # print("distances = ",distances)
    # indmin = np.where(distances==distances.min())
    # # print("indmin = ",indmin," min = ",distances.min())
    # u0 = np.array( [ aa[indmin[0][0]], bb[indmin[0][0]], aa[indmin[1][0]], bb[indmin[1][0]] ])
    # print("u0 = ",u0)

    num = 4
    binit1 = np.concatenate( (np.array(create_binit_xy(num, s1)),np.array(create_binit_xy(num, s1))) )
    ainit1_yz = np.array( create_ainit_yz(num, s1) )
    ainit1_xz = np.array( create_ainit_xz(num, s1) )
    ainit1 = np.concatenate((ainit1_yz,ainit1_xz))
    agrid1, bgrid1 = np.meshgrid(ainit1, binit1)
    agrid1 = agrid1.reshape((-1,))
    bgrid1 = bgrid1.reshape((-1,))
    points1 = s1.surface_pt(agrid1,bgrid1)
    normals1 = s1.surface_normal(agrid1,bgrid1)
    binit2 =  np.concatenate( (np.array(create_binit_xy(num, s2)),np.array(create_binit_xy(num, s2))) )
    ainit2_yz = np.array( create_ainit_yz(num, s2) )
    ainit2_xz = np.array( create_ainit_xz(num, s2) )
    ainit2 = np.concatenate((ainit2_yz,ainit2_xz))
    agrid2, bgrid2 = np.meshgrid(ainit2, binit2)
    agrid2 = agrid2.reshape((-1,))
    bgrid2 = bgrid2.reshape((-1,))
    points2 = s2.surface_pt(agrid2,bgrid2)
    normals2 = s2.surface_normal(agrid2,bgrid2)
    distances = np.zeros((points1.shape[0],points2.shape[0]))
    for i in range(points1.shape[0]):
        for j in range(points2.shape[0]):
            distances[i,j] = np.linalg.norm(points1[i,:]-points2[j,:])
    # print(distances)
    indmin = np.where(distances==distances.min())
    print("indmin = ",indmin)
    u0 = np.array( [ agrid1[indmin[0][0]], bgrid1[indmin[0][0]], agrid2[indmin[1][0]], bgrid2[indmin[1][0]] ])
    print(u0)
    pv_pts_ext_s1 = pv.PolyData(points1)
    pv_pts_ext_s1["normal"] = normals1
    pv_pts_ext_s2 = pv.PolyData(points2)
    pv_pts_ext_s2["normal"] = normals2
    p.add_mesh(pv_pts_ext_s1.glyph(factor=0.005, geom=pv.Sphere()),color="yellow")
    p.add_mesh(pv_pts_ext_s1.glyph(orient="normal",factor=0.02, geom=pv.Arrow()),color="yellow")
    p.add_mesh(pv_pts_ext_s2.glyph(factor=0.005, geom=pv.Sphere()),color="yellow")
    p.add_mesh(pv_pts_ext_s2.glyph(orient="normal",factor=0.02, geom=pv.Arrow()),color="yellow")

    pv_start_s1 = pv.PolyData(s1.surface_pt(agrid1[indmin[0][0]], bgrid1[indmin[0][0]]))
    pv_start_s1["normal"] = s1.surface_normal([agrid1[indmin[0][0]]], [bgrid1[indmin[0][0]]])
    pv_start_s2 = pv.PolyData(s2.surface_pt(agrid2[indmin[1][0]], bgrid2[indmin[1][0]]))
    pv_start_s2["normal"] = s2.surface_normal([agrid2[indmin[1][0]]], [bgrid2[indmin[1][0]]])
    p.add_mesh(pv_start_s1.glyph(factor=0.01, geom=pv.Sphere()),color="green")
    p.add_mesh(pv_start_s1.glyph(orient="normal",factor=0.03, geom=pv.Arrow()),color="green")
    p.add_mesh(pv_start_s2.glyph(factor=0.01, geom=pv.Sphere()),color="green")
    p.add_mesh(pv_start_s2.glyph(orient="normal",factor=0.03, geom=pv.Arrow()),color="green")

    pv_startall_s1 = pv.PolyData(s1.surface_pt(agrid1[indmin[0]], bgrid1[indmin[0]]))
    pv_startall_s1["normal"] = s1.surface_normal(agrid1[indmin[0]], bgrid1[indmin[0]])
    pv_startall_s2 = pv.PolyData(s2.surface_pt(agrid2[indmin[1]], bgrid2[indmin[1]]))
    pv_startall_s2["normal"] = s2.surface_normal(agrid2[indmin[1]], bgrid2[indmin[1]])
    p.add_mesh(pv_startall_s1.glyph(factor=0.008, geom=pv.Sphere()),color="orange")
    p.add_mesh(pv_startall_s1.glyph(orient="normal",factor=0.008, geom=pv.Arrow()),color="orange")
    p.add_mesh(pv_startall_s2.glyph(factor=0.008, geom=pv.Sphere()),color="orange")
    p.add_mesh(pv_startall_s2.glyph(orient="normal",factor=0.008, geom=pv.Arrow()),color="orange")

    # p.show_grid()
    # p.show_bounds()
    # p.camera_position = 'xy'
    # p.camera.SetParallelProjection(True)
    # p.show(window_size=(1600,900))

    # sys.exit()

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
        while (np.linalg.norm(f(x+t*d,s1,s2)) > np.linalg.norm(f(x,s1,s2)))and (t>0.01) :
            t -= 0.01
        return t
    cc = 0
    itermax = 4000
    u = u0.copy()
    dk = np.ones(u.shape)

    # ### test pour comparaison avec le C++
    # print("s1 : xc=",s1.xc," yc=",s1.yc," zc=",s1.zc," rx=",s1.rx," ry=",s1.ry," rz=",s1.rz," e=",s1.e," n=",s1.n," q=",s1.q)
    # print("s2 : xc=",s2.xc," yc=",s2.yc," zc=",s2.zc," rx=",s2.rx," ry=",s2.ry," rz=",s2.rz," e=",s2.e," n=",s2.n," q=",s2.q)
    # print("u0 = ",u0)
    # val1 = np.array([0.43])
    # val2 = np.array([0.65])
    # # print("s1.surface_pt(val1,val2)      = ",s1.surface_pt(val1,val2),     " s2.surface_pt(val2,val1)      = ",s2.surface_pt(val2,val1))
    # # print("s1.surface_normal(val1,val2)  = ",s1.surface_normal(val1,val2), " s2.surface_normal(val2,val1)  = ",s2.surface_normal(val2,val1))
    # # print("s1.surface_tangent(val1,val2) = ",s1.surface_tangent(val1,val2)," s2.surface_tangent(val2,val1) = ",s2.surface_tangent(val2,val1))
    # # print("u=",u)
    # # print("grad_f_contacts(u,s1,s2) = ",grad_f_contacts(u,s1,s2), " f_contacts(u,s1,s2) = ",f_contacts(u,s1,s2))

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
    # sys.exit()

    node_final_pt1 = pv.PolyData(final_pt1)
    node_final_pt1["normal"] = normal_final_pt1
    glyphs_final_pt1 = node_final_pt1.glyph(factor=0.005, geom=pv.Sphere())
    p.add_mesh(glyphs_final_pt1,color="yellow")
    glyphs_normal_final_pt1 = node_final_pt1.glyph(orient="normal",factor=0.1, geom=pv.Arrow())
    p.add_mesh(glyphs_normal_final_pt1,color="yellow")


    node_final_pt2 = pv.PolyData(final_pt2)
    node_final_pt2["normal"] = normal_final_pt2
    glyphs_final_pt2 = node_final_pt2.glyph(factor=0.005, geom=pv.Sphere())
    p.add_mesh(glyphs_final_pt2,color="yellow")
    glyphs_normal_final_pt2 = node_final_pt2.glyph(orient="normal",factor=0.1, geom=pv.Arrow())
    p.add_mesh(glyphs_normal_final_pt2,color="yellow")

    line_points = np.row_stack((final_pt1, final_pt2))
    line = pv.PolyData(line_points)
    cells = np.full((len(line_points)-1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(line_points)-1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(line_points), dtype=np.int_)
    line.lines = cells
    line["scalars"] = np.arange(line.n_points)
    tube = line.tube(radius=0.001)
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
