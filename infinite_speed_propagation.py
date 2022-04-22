import numpy as np
from numpy import sin, cos, exp, pi
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
# plt.rc('figure',  figsize=(16/1.5, 9/1.5))
plt.style.use('ggplot')


""" SETTING EVERYTHING UP (parameters for the model) """
N = 100  # inner points for the space mesh
T = 0.1  # final time


a = 1  # diffusion coefficient
dx = 1 / (N + 1)  # space step
dt = 1e-5
nu = a * dt / (dx*dx)
theta = 0.5

M = int(1 / dt)

x_mesh = np.linspace(0, 1, N+2)  # mesh in space
x_inner_mesh = x_mesh[1:N+1]


# initial condition of the Cauchy Problem
def init_cond(x): return np.int64(x > 0.25) - np.int64(x > 0.75)

""" --- end of setup --- """


u_0 = init_cond(x_inner_mesh)  # initial conditions
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

""" IMPLEMENTATION OF THETA METHOD """

cols = {0: 'red', 0.5: 'blue', 1: 'black'}

# compute solution
for j in range(int(T/dt)):
    if (j % 10 == 0):
        print("check", j, "out of ", T/dt)
    u_next = spsolve(matrix_bh, matrix_ch @ u_list[j][1:-1])
    u_next = np.concatenate(([0], u_next, [0]))
    u_list.append(u_next)
""" --- end of computing solutions --- """


plt.plot(x_mesh, u_list[0], marker='o', markevery=N//2, ls="--", c="blue", label=rf"$t$ = 0")
plt.plot(x_mesh, u_list[int(0.01/dt)], marker='o', markevery=N//2, ls="--", c="red", label=rf"$t = {int(0.00010 / dt)} \delta t$")
print("nu =", nu)

""" PLOTTING STUFF """

plt.legend()
plt.title(rf"Graphes de la solution aux temps $t=0$ et $t=10 \delta t$ ({10*dt})")
plt.show()
""" --- end of plotting stuff --- """
