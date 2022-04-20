import numpy as np
from numpy import sin, cos, exp, pi
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
plt.style.use('ggplot')


""" SETTING EVERYTHING UP (parameters for the model) """
T = 0.5  # final time


a = 1  # diffusion coefficient
nu = 0.5
# inner points for the space mesh
Ns = np.array([int(1e1), int(1e2), int(1e3)])
errors_list = []

thetas = [0, 1, 0.5]  # method chosen

for theta in thetas:
    errors = []

    dxs = []
    dts = []
    for N in Ns:
        dx = 1 / (N + 1)  # space step
        dt = nu * dx * dx / a

        dxs.append(dx)
        dts.append(dt)

        M = int(1 / dt)

        x_mesh = np.linspace(0, 1, N+2)  # mesh in space
        x_inner_mesh = x_mesh[1:N+1]
        t_mesh = np.linspace(0, T, M)  # mesh in time

        # initial condition of the Cauchy Problem
        def init_cond(x, k): return sin(k*pi*x)
        """ --- end of setup --- """

        """ TRUE SOLUTION """

        k = 1  # parametrize k in initial condition
        def u_exact(t, x, k): return exp(-k*k*pi*pi*t) * sin(k*pi*x)
        u_sol = u_exact(0.5, x_mesh, k)

        """ --- end of true solution --- """

        """ IMPLEMENTATION OF THETA METHOD """
        u_0 = init_cond(x_inner_mesh, k)  # initial conditions
        u_0 = np.concatenate(([0], u_0, [0]))  # init cond. + border conditions

        # Create Bh := Tridiag(-theta * nu, 1 + 2 * theta * nu, -theta * nu)_n
        matrix_bh = sp.spdiags(
            [[-theta * nu for _ in range(N)],
             [1 + 2 * theta * nu for _ in range(N)],
             [- theta * nu for _ in range(N)]],
            (-1, 0, 1),
            N,
            N,
            format='csc'
        )

        # create Ch := Tridiag((1-theta) * nu , 1 - 2 * (1 - theta) * nu, (1-theta) * nu)_n
        matrix_ch = sp.spdiags(
            [[(1 - theta) * nu for _ in range(N)],
             [1 - 2 * (1 - theta) * nu for _ in range(N)],
             [(1 - theta) * nu for _ in range(N)]],
            (-1, 0, 1),
            N,
            N,
            format='csc'
        )

        u_list = [u_0]

        # compute solution with
        print(int(T/dt))
        for j in range(int(T/dt)):
            if (j % 100 == 0):
                print("check", j, "out of ", T/dt)
            u_next = spsolve(matrix_bh, matrix_ch @ u_list[j][1:-1])
            u_next = np.concatenate(([0], u_next, [0]))
            u_list.append(u_next)

        """ --- end of computing solutions --- """

        e_h = u_sol - u_list[-1]
        errors.append(np.max(np.abs(e_h)))  # error L^infty
        # errors.append(np.sqrt(dx) * np.linalg.norm(e_h, 2)) # errors L^2 discrete

        '''
        plt.plot(u_list[-1], ls='--', lw=2, c='blue', label="Approx.")
        plt.plot(u_sol, lw=0.7, c='blue', label="Exact.")
        plt.legend()
        plt.show()
        '''

    errors_list.append(errors)


dxs = np.array(dxs)
dts = np.array(dts)

print(errors_list)

methods = ["Explicit", "Implicit", "Crank-Nicolson"]

plt.loglog(dxs, dxs, marker='o', label="O(dx)")
plt.loglog(dxs, dxs*dxs, marker='o', label="O(dx^2)")
for j in range(len(errors_list)):
    plt.loglog(dxs, errors_list[j], marker='o', ls='--', label=methods[j])
plt.xlabel(r'$\delta x$')
plt.ylabel('')
plt.title(r"Calcul de l'erreur en norme ${L}^{\infty}$")
plt.legend()
plt.show()
