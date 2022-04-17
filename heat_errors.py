import numpy as np
from numpy import sin, cos, exp, pi
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
plt.style.use('ggplot')



""" SETTING EVERYTHING UP (parameters for the model) """ 
T = 0.5 # final time


a = 1 # diffusion coefficient
nu = 10
Ns = np.array([10, 100, 1000]) # inner points for the space mesh
dxs = []
dts = []

errors = []

print(dts)
theta = 0.5 # method chosen

for N in Ns:
    dx = 1 / (N + 1) # space step
    dt = nu * dx * dx / a

    dxs.append(dx)
    dts.append(dt)

    M = int(1 / dt)

    x_mesh = np.linspace(0, 1, N+2) # mesh in space
    x_inner_mesh = x_mesh[1:N+1]
    t_mesh = np.linspace(0, T, M) # mesh in time

    init_cond = lambda x, k: sin(k*pi*x) # initial condition of the Cauchy Problem 

    """ --- end of setup --- """

        
    """ TRUE SOLUTION """

    k = 1 # parametrize k in initial condition
    u_exact = lambda t, x, k: exp(-k*k*pi*pi*t) * sin(k*pi*x)
    u_sol = u_exact(0.5, x_mesh, k)

    """ --- end of true solution --- """


    """ IMPLEMENTATION OF THETA METHOD """
    u_0 = init_cond(x_inner_mesh, k) # initial conditions
    u_0 = np.concatenate(([0], u_0, [0])) # init cond. + border conditions

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

    # compute solution
    print(int(T/dt))
    for j in range(int(T/dt)):
        # if (j % 100 == 0) : print("check", j,"out of ", T/dt)
        u_next = spsolve(matrix_bh, matrix_ch @ u_list[j][1:-1])
        u_next = np.concatenate(([0], u_next, [0]))
        u_list.append(u_next)

    """ --- end of computing solutions --- """

    e_h = u_sol - u_list[-1]
    errors.append(np.max(np.abs(e_h)))

    plt.plot(u_list[-1], ls='--', lw=2)
    plt.plot(u_sol, lw=0.7)
    plt.show()



dxs = np.array(dxs)
dts = np.array(dts)

plt.loglog(dxs, dxs, label="O(dx)")
plt.loglog(dxs, dxs*dxs, label="O(dx^2)")
plt.loglog(dxs, errors, ls='--', label="numerical method")
plt.legend()
plt.show()