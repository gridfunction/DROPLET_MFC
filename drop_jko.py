# Approx. JKO scheme for lubrication model
# Apply Newton's method for critical point system
#
# dim = 1: 1D simulation
# dim = 2: 2D simulation

from ngsolve import *
from ngsolve.meshes import MakeQuadMesh, Make1DMesh
from ngsolve.solvers import Newton
import numpy as np

import os
os.makedirs('NGS_data', exist_ok=True)

# switch between dim =1 or dim = 2
# take dt = 0.0001 or dt = 0.001
dim = 1 # do not change
dt = 0.0001

per = True
# parameters
Ps  = 0.5
eps = 0.3
gma = 0.04
alpha = 0.01
tend = 0.4
nt = int(0.01/dt+0.5)

t = 0.0

if dim==1:
    hinit = 1-0.2*cos(2*pi*x)
    mesh = Make1DMesh(n=32, periodic=per)
elif dim==2:
    hinit = 1+0.2*cos(2*pi*x)*cos(2*pi*y)
    mesh = MakeQuadMesh(nx=32, ny=32, periodic_x=per, periodic_y=per)

def Pi(h):
    return eps**2/h**3*(1.0-eps/h) - Ps

def V1(h):
    return h**3
def V2(h):
    return gma/(h+0.1)

def dV1(h):
    return 3*h**2
def dV2(h):
    return -gma/(h+0.1)**2


if per == True:
    V = Periodic(H1(mesh, order=4))
    if dim==1:
        S = Periodic(H1(mesh, order=4))
    else:
        S = Periodic(HDiv(mesh, order=3))
else:
    V = H1(mesh, order = 4)
    if dim==1:
        S = H1(mesh, order=4, dirichlet=".*")
    else:
        S = HDiv(mesh, order=3, dirichlet=".*")
W = L2(mesh, order=3)
if dim==1:
    W2 = L2(mesh, order=3)
else:
    W2 = VectorL2(mesh, order=3)
# unknowns: h, m, s, n, phi, sigma
fes = W*W2*W*W2*V*S
(h, m, s, n, phi, sigma), (th, tm, ts, tn, tphi, tsigma) = fes.TnT()
gfu0 = GridFunction(fes)
h0 = gfu0.components[0]
gfu1 = GridFunction(fes)
h1 = gfu1.components[0]

gfu = GridFunction(fes)
hh = gfu.components[0]

a = BilinearForm(fes, condense=True)
if dim==1:
    a += ((m/V1(h)-grad(phi))*tm \
            + (s/V2(h)-phi)*ts \
            +n*tsigma + alpha*h*grad(tsigma) \
            +(dt*n-sigma)*tn \
            +((h-h0-s)*tphi - m*grad(tphi)) \
            +(phi-alpha*grad(sigma)+dt*Pi(h)-m*m/2/V1(h)**2*dV1(h)
                -s*s/2/V2(h)**2*dV2(h))*th)*dx
else:
    a += ((m/V1(h)-grad(phi))*tm \
            + (s/V2(h)-phi)*ts \
            +n*tsigma + alpha*h*div(tsigma) \
            +(dt*n-sigma)*tn \
            +((h-h0-s)*tphi - m*grad(tphi)) \
            +(phi-alpha*div(sigma)+dt*Pi(h)-m*m/2/V1(h)**2*dV1(h)
                -s*s/2/V2(h)**2*dV2(h))*th)*dx


# initialize data
hh.Set(hinit)
h0.vec.data = hh.vec

vtk = VTKOutput(mesh,coefs=[hh],names=["height"],filename="NGS_data/drop_jko"+str(dim)+str(nt),
        subdivision=3)
vtk.Do()

xGrid = np.linspace(0, 1, 201)

hList = []
count = 0
with TaskManager():
    while t < tend+dt/2:
        t += dt
        count += 1
        zz = Newton(a, gfu, printing=False)
        h0.vec.data = hh.vec
        if count%nt == 0 :
            if dim==2:
               vtk.Do()
            if dim==1:
                hGrid = np.array([hh(x) for x in xGrid])
                hList.append(hGrid)

            print(t, count)
        print('\r Time: %.2e, count: %4i, iter: %4i'%(t, count, zz[1]), end="")

if dim==1:
    hList = np.array(hList)
    print('\n', nt)
    np.savetxt('NGS_data/jko1D'+str(nt)+'.txt', hList)

