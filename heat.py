from firedrake import *
from firedrake.petsc import PETSc
print = lambda x: PETSc.Sys.Print(x)
from ufl.algorithms.ad import expand_derivatives

import numpy
from datetime import datetime
import pprint

t = Constant(0.0)
dt = Constant(1.0)


class Problem(object):
    def __init__(self, N, degree):
        super().__init__()
        self.N = N
        self.degree = degree


    def mesh(self):
        mesh = RectangleMesh(self.N, self.N, 1, 1)
        return mesh

    def function_space(self, mesh):
        Ve = FiniteElement("CG", mesh.ufl_cell(), self.degree)
        Pe = FiniteElement("R", mesh.ufl_cell(), 0)
        Ze = MixedElement([Ve, Pe])
        return FunctionSpace(mesh, Ze)

    def initial_condition(self, Z):
        (x,y) = SpatialCoordinate(Z.mesh())
        (u0,p0) = self.exact_soln(Z.mesh())
        z = Function(Z)
        z.sub(0).interpolate(u0)
        z.sub(1).interpolate(p0)
        return z

    def exact_soln(self, mesh):
        (x,y) = SpatialCoordinate(mesh)
        u =e**(2*(t+t**2)+x+y)
        p = exp(pi*t)+t**3
        return (u, p)

    def function_space(self, mesh):
        Ve = FiniteElement("CG", mesh.ufl_cell(), self.degree)
        Pe = FiniteElement("R", mesh.ufl_cell(), 0)
        Ze = MixedElement([Ve, Pe])
        return FunctionSpace(mesh, Ze)


    def BC(self, Z):
        mesh = Z.mesh()
        (u,p) = self.exact_soln(mesh)
        bcs = [
               DirichletBC(Z.sub(0), u, "on_boundary"),
              ]
        return bcs

    def Rhs(self, mesh):
        (u_exact,p_exact) = self.exact_soln(mesh)
        u_rhs = expand_derivatives(diff(u_exact, t))- p_exact*div(grad(u_exact))
        p_rhs = expand_derivatives(diff(p_exact, t))
        return (u_rhs, p_rhs)


    def form(self, z, test_z, Z):
        (u, p) = split(z)
        (v, q) = split(test_z)
        (u_rhs, p_rhs) = self.Rhs(Z.mesh())
        F = p*inner(grad(u), grad(v))*dx- inner(u_rhs, v)*dx- inner(p_rhs, q)*dx
        return F

	

if __name__ == "__main__":
    N = 20
    problem = Problem(N, 1)
    mesh = problem.mesh()
    Z = problem.function_space(mesh)
    bcs = problem.BC(Z)
    z0 = problem.initial_condition(Z)
    z = Function(Z)
    z_test = TestFunction(Z)


    sp = {
        "mat_type": "matfree",
        "snes_type": "newtonls",
        "snes_monitor": None,
#        "snes_converged_reason": None,
        "snes_linesearch_type": "basic",
        "ksp_type": "fgmres",
#        "ksp_monitor_true_residual": None,
        "ksp_max_it": 10,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
        "pc_fieldsplit_0_fields": "0",
        "pc_fieldsplit_1_fields": "1",
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "python",
        "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
        "fieldsplit_0_assembled_pc_type": "lu",
        "fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps",
        "fieldsplit_0_assembled_mat_mumps_icntl_14": 200,
#        "fieldsplit_0_ksp_converged_reason": None,

        "mat_mumps_icntl_14": 200,
        "fieldsplit_1_ksp_type": "gmres",
#        "fieldsplit_1_ksp_converged_reason": None,
#        "fieldsplit_1_ksp_monitor_true_residual": None,
        "fieldsplit_1_ksp_max_it": 1,
        "fieldsplit_1_ksp_convergence_test": "skip",
        "fieldsplit_1_pc_type": "none",

             }

    T  = 1/N
    dt = 1/N
    F_euler = (
         inner(split(z)[0], split(z_test)[0])*dx
       - inner(split(z0)[0],split(z_test)[0])*dx
       + inner(split(z)[1], split(z_test)[1])*dx
       - inner(split(z0)[1], split(z_test)[1])*dx
       + dt*(problem.form(z,z_test,Z))
      )

    nvproblem = NonlinearVariationalProblem(F_euler, z, bcs=bcs)
    solver = NonlinearVariationalSolver(nvproblem, solver_parameters = sp)

    while (float(t) < T):
        t.assign(float(t) + float(dt))
        (U,P) = problem.exact_soln(mesh)
        (u,p) = z.split()
        p.interpolate(P)
#        u.interpolate(U)
        solver.solve()
        z0.assign(z)

#    A = assemble(inner(U - split(z)[0], U - split(z)[0])*dx)/ assemble(inner(U, U)*dx)
#    print(sqrt(A))

#    B = assemble(inner(P - split(z)[1], P - split(z)[1])*dx)
#    print(sqrt(B))


