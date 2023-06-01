# -*- coding: utf-8 -*-
"""
@author: hgokc
"""
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# Set log level
set_log_level(LogLevel.WARNING)

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True,
               "eliminate_zeros": True,
               "precompute_basis_const": True,
               "precompute_ip_const": True}

# Create mesh and define function space
mesh = Mesh()
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile("mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile("mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
print(mesh.topology().dim() - 1)
File("MSH.pvd") << mesh
File("MSH2.pvd") << mf
boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 0)
class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)
b_c = Boundary()
b_c.mark(boundary_markers, 3)
File("MSH3.pvd") << boundary_markers

# Compiling subdomains
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
dx_filled = Measure("dx", domain=mesh, subdomain_data=mf, subdomain_id=2)
dx_main = Measure("dx", domain=mesh, subdomain_data=mf, subdomain_id=1)
V = VectorFunctionSpace(mesh, "CG", 2)  # Lagrange
W_tensor = TensorFunctionSpace(mesh, "CG", 2)

def boundary_bot(x, on_boundary):
    tol = 1E-7
    return on_boundary and near(x[1], 0, tol)
bc_bot = DirichletBC(V, [0, 0, 0], boundary_bot)

def boundary_top(x, on_boundary):
    tol = 1E-7
    return on_boundary and near(x[1], 10.0, tol)

max_disp = -2.75
disp = Constant([0.0, 0.0, 0.0])
bc_top = DirichletBC(V, disp, boundary_top)
bcs = [bc_bot, bc_top]
commit = input('commit:')

# Define functions
du = TrialFunction(V)  # Incremental displacement
v = TestFunction(V)  # Test function
u = Function(V)  # Displacement from previous iteration
boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_subdomains.set_all(0)
AutoSubDomain(boundary_top).mark(boundary_subdomains, 1)
dss = ds(subdomain_data=boundary_subdomains)

# Kinematics
d = u.geometric_dimension()  # Space dimension
I = Identity(d)  # Identity tensor
F = variable(I + grad(u))  # Deformation gradient
C = F.T * F  # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J = det(F)


# Elasticity parameters
E1, nu1 = 5*10e-10, 0.0  #void
mu1, lmbda1 = Constant(E1 / (2 * (1 + nu1))), Constant(E1 * nu1 / ((1 + nu1) * (1 - 2 * nu1)))
E2, nu2 = 26.0, 0.25  # main
mu2, lmbda2 = Constant(E2 / (2 * (1 + nu2))), Constant(E2 * nu2 / ((1 + nu2) * (1 - 2 * nu2)))

# Stored strain energy density (compressible neo-Hookean model)
psi_filled = (mu1 / 2) * (Ic - 3) - mu1 * ln(J) + (lmbda1 / 2) * (ln(J)) ** 2
psi_main = (mu2 / 2) * (Ic - 3) - mu2 * ln(J) + (lmbda2 / 2) * (ln(J)) ** 2
psi = psi_filled + psi_main

# Total potential energy
Pi_filled = psi_filled * dx_filled
Pi_main = psi_main * dx_main

# Compute first variation of Pi (directional derivative about u in the direction of v)
dPi_filled = derivative(Pi_filled, u, v)
dPi_main = derivative(Pi_main, u, v)
dPi = dPi_filled + dPi_main

# Compute Jacobian of F
Jac_filled = derivative(dPi_filled, u, du)
Jac_main = derivative(dPi_main, u, du)
Jac = Jac_filled + Jac_main

# Piola-Kirchhoff Stress
stress = diff(psi, F)

# Solve variational problem
store_u = np.linspace(0, max_disp, 100)
Traclist = []
normal = Constant((0, 1, 0))
file = File("./Results " + "max_disp=" + str(max_disp) +  "--" + str(
        commit) + "/displacement.pvd");
file2 = File("./Results " + "max_disp=" + str(max_disp) +  "--" + str(
        commit) + "/stress.pvd");
for d in store_u:
    print(d)
    disp.assign(Constant([0.0, d, 0.0]))
    solve(dPi == 0, u, bcs,
    form_compiler_parameters=ffc_options)
    file << u
    stress_calculation = (project((stress), W_tensor))
    file2 << stress_calculation

    # storedtraction = assemble(dot(dot(stress, normal), normal) * dss(1))  # force
    traction = dot(stress, normal)
    # storedproject = project(traction, V_vector)
    storedtraction = assemble((traction[1]) * dss(1))
    Traclist.append(storedtraction)

print(store_u)
print(Traclist)

  
  
# Plot
K = -1
store_u = [x * K for x in store_u]
Traclist = [y * K for y in Traclist]
plt.plot(store_u, Traclist)
plt.xlabel("Displacement")
plt.ylabel("Force")
plt.title("Graph for the displacement" + str(max_disp))
plt.savefig('./Results ' + 'max_disp=' + str(max_disp) +  '--'+ str(
        commit) + '/Graph.png')
print("Done")
