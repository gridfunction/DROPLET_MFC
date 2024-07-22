# FEM for lubrication model
# with BDF2 time stepping
# dim = 1: 1D simulation
# dim = 2: 2D simulation

from ngsolve import *
from ngsolve.meshes import MakeQuadMesh, Make1DMesh
from ngsolve.solvers import Newton
import numpy as np

import os
os.makedirs('NGS_data', exist_ok=True)

# switch between dim = 1 or dim = 2
dim = 1

per = True # periodic domain

# parameters
Ps  = 0.5
eps = 0.3
gma = 0.04
alpha = 0.01
tend = 0.4
dt = 0.0001
nt = int(0.01/dt+0.5)

t = 0.0

if dim==1:
    hinit = 1-0.2*cos(2*pi*x)
    mesh = Make1DMesh(n=32, periodic=per)
elif dim==2:
    hinit = 1+0.2*cos(2*pi*x)*cos(2*pi*y)
    mesh = MakeQuadMesh(nx=32, ny=32, periodic_x=per, periodic_y=per)

def dU(h):
    return eps**2/h**3*(1.0-eps/h) - Ps

def V1(h):
    return h**3
def V2(h):
    return gma/(h+0.1)


if per == True:
    V = Periodic(H1(mesh, order=4))
else:
    V = H1(mesh, order = 4)

fes = V*V

(h, p), (v, q) = fes.TnT()
gfu0 = GridFunction(fes)
h0, p0 = gfu0.components
gfu1 = GridFunction(fes)
h1, p1 = gfu1.components

gfu = GridFunction(fes)
hh, ph = gfu.components

a = BilinearForm(fes, condense=True)
a += (h-h0)/dt*v*dx + V1(h)*grad(p)*grad(v)*dx \
        + V2(h)*p*v*dx \
        + (p-dU(h))*q*dx - alpha**2* grad(h)*grad(q)*dx

a2 = BilinearForm(fes, condense=True)
a2 += (3*h-4*h0+h1)/2/dt*v*dx + V1(h)*grad(p)*grad(v)*dx \
        + V2(h)*p*v*dx \
        + (p-dU(h))*q*dx - alpha**2*grad(h)*grad(q)*dx

# initialize data
hh.Set(hinit)
h0.vec.data = hh.vec

vtk = VTKOutput(mesh,coefs=[hh],names=["height"],filename="NGS_data/drop"+str(dim),subdivision=3)
vtk.Do()

xGrid = np.linspace(0, 1, 201)

hList = []
count = 0
with TaskManager():
    while t < tend+dt/2:
        t += dt
        count += 1
        if count==1: # BDF1
            Newton(a, gfu, printing=False)
        else:
            Newton(a2, gfu, printing=False)

        h1.vec.data = h0.vec
        h0.vec.data = hh.vec
        if count%nt == 0 :
            if dim==2:
               vtk.Do()
            if dim==1:
                hGrid = np.array([hh(x) for x in xGrid])
                hList.append(hGrid)

            print(t, count)
        print('\r Time: %.2e, count: %4i'%(t, count), end="")

if dim==1:
    hList = np.array(hList)
    print('\n', nt)
    np.savetxt('NGS_data/fem1D.txt', hList)

