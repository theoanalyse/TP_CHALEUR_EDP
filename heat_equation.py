import numpy as np
from numpy import sin, cos, exp, pi
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve



""" SETTING EVERYTHING UP (parameters for the model) """
N = 100 # inner points for the space mesh 
T = 1 # final time


a = 1 # diffusion coefficient
dx = 1 / (N + 1) # space step
dt = 1e-5
nu = a * dt / (dx*dx)
theta = 0.5

print("nu =", nu)

M = int(1 / dt)

x_mesh = np.linspace(0, 1, N+2) # mesh in space
x_inner_mesh = x_mesh[1:N+1]
t_mesh = np.linspace(0, T, M) # mesh in time

init_cond = lambda x, k: sin(k*pi*x) # initial condition of the Cauchy Problem 

""" --- end of setup --- """

""" IMPLEMENTATION OF THETA METHOD """
k = 1 # parametrize k in initial condition
u_0 = init_cond(x_inner_mesh, k) # initial conditions
u_0 = np.concatenate(([0],u_0, [0])) # init cond. + border conditions

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

energy = [(np.sqrt(dx) * np.linalg.norm(u_0, 2))**2]

# compute solution
for j in range(int(T/dt)):
    if (j % 1000 == 0) : print("check", j,"out of ", T/dt)
    u_next = spsolve(matrix_bh, matrix_ch @ u_list[j][1:-1])
    u_next = np.concatenate(([0], u_next, [0]))
    u_list.append(u_next)
    
    energy.append((np.sqrt(dx) * np.linalg.norm(u_next, 2))**2)

""" --- end of computing solutions --- """



""" TRUE SOLUTION """

u_exact = lambda t, x, k: exp(-k*k*pi*pi*t) * sin(k*pi*x)

u_sol_1 = u_exact(0.1, x_mesh, k)
u_sol_2 = u_exact(0.5, x_mesh, k)

""" --- end of true solution --- """


""" PLOTTING STUFF """
plt.style.use('ggplot')
plt.plot(x_mesh, u_list[int(0.1/dt) + 1], c='blue',ls='--', lw=2, label="approx T=0.1")
plt.plot(x_mesh, u_sol_1, c='blue', lw=0.8, label="exact T=0.1")
plt.plot(x_mesh, u_list[-1], ls='--', c='red', lw=2, label="approx T=0.5")
plt.plot(x_mesh, u_sol_2, c='red', lw=0.8, label="exact T=0.5")
plt.title(r"Approximation de la solution exacte par le $\theta$-schema")
plt.legend()
plt.show()



plt.plot(np.linspace(0, 1, int(T/dt)), np.array([0.5 * exp(- 2 * k**2 * pi **2 * n * dt) for n in range(int(T/dt))]), c='red', ls='-', lw=0.8, label='Exact.')
plt.plot(np.linspace(0, 1, int(T/dt) + 1), energy, marker='o', markevery=10000, c='blue', ls='--', dashes=(3, 3), lw = 2, label="Approx.")
plt.xlabel(r'$t (temps)$')
plt.ylabel(r'$\mathcal{E}^n$ vs. $E(t)$ (energie)')
plt.legend()
plt.title(r"Énergie de la solution $\mathbf{u}_h$ en fonction du temps")
plt.show()
""" --- end of plotting stuff --- """