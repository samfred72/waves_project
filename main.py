import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def neumann_BC_y(psi, f=0, g=0, dx=0.01):
    psi[0, :] = psi[2, :] - 2 * dx * f
    psi[-1, :] = psi[-3, :] - 2 * dx * g
    return psi


def neumann_BC_x(psi, f=0, g=0, dx=0.01):
    psi[:, 0] = psi[:, 2] - 2 * dx * f
    psi[:, -1] = psi[:, -3] - 2 * dx * g
    return psi


def dirichlet_BC_y(psi, f=0, g=0, dx=0.01):
    psi[0, :] = f
    psi[-1, :] = g
    return psi


def dirichlet_BC_x(psi, f=0, g=0, dx=0.01):
    psi[:, 0] = f
    psi[:, -1] = g
    return psi


def absorbing_BC_x(psi, psi_prev, r):
    psi[:, 0] = psi_prev[:, 1] + ((r - 1) / (r + 1)) * (psi_prev[:, 2] - psi[:, 1])
    psi[:, -1] = psi_prev[:, -2] + ((r - 1) / (r + 1)) * (psi_prev[:, -1] - psi[:, -2])
    return psi


def absorbing_BC_y(psi, psi_prev, r):
    psi[0, :] = psi_prev[1, :] + ((r - 1) / (r + 1)) * (psi_prev[2, :] - psi[1, :])
    psi[-1, :] = psi_prev[-2, :] + ((r - 1) / (r + 1)) * (psi_prev[-1, :] - psi[-2, :])
    return psi


def x_boundary_conditions(psi, psi_prev=None, xtype='n', xf=0, xg=0, dx=0, r=3):
    if xtype == 'n':
        return neumann_BC_x(psi, xf, xg, dx)
    elif xtype == 'd':
        return dirichlet_BC_x(psi, xf, xg, dx)
    elif xtype == 'a':
        return absorbing_BC_x(psi, psi_prev, r)
    raise Exception("xtype and ytype must be either 'n' or 'd' or 'a'")


def y_boundary_conditions(psi, psi_prev=None, ytype='n', yf=0, yg=0, dy=0, r=3):
    if ytype == 'n':
        return neumann_BC_y(psi, yf, yg, dy)
    elif ytype == 'd':
        return dirichlet_BC_y(psi, yf, yg, dy)
    elif ytype == 'a':
        return absorbing_BC_y(psi, psi_prev, r)
    raise Exception("xtype and ytype must be either 'n' or 'd'")


def n_slit_barrier(x,y,position,n_slits,slit_dims):
    horiz_mask = (position < x[0,:]).astype(int)
    horiz_mask *= (x[0,:] < position + slit_dims[0]).astype(int)
    vert_mask = np.heaviside((y - 1/(n_slits+1))+slit_dims[1],1) *\
            np.heaviside(-(y - 1/(n_slits+1))+slit_dims[1],1)
    for i in range(1,n_slits):
        vert_mask += np.heaviside((y - (i+1)/(n_slits+1))+slit_dims[1],1) *\
            np.heaviside(-(y - (i+1)/(n_slits+1))+slit_dims[1],1)
    return np.logical_not((np.logical_not(vert_mask) * horiz_mask)).astype(int)


if __name__ == "__main__":
    nx = 501
    ny = 501
    velocity = 1
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    grid_x, grid_y = np.meshgrid(x, y)
    dx = 1 / nx
    dy = 1 / ny
    dt = dx / 2

    # Initial condition options
    gaussian = True
    standing = False
    droplet = False
    standing_drop = False

    u_0 = np.zeros([nx,ny])
    u_1 = np.zeros_like(u_0)
    if gaussian:
        u_0 = np.exp(-(10 * (grid_x - 0.2)) ** 2) * np.cos(100 * grid_x)
        u_1 = np.exp(-(10 * (grid_x - 0.2 + velocity * dt)) ** 2) * np.cos(100 * (grid_x + velocity * dt))
    elif standing:
        u_0 = np.heaviside(-grid_x, 1)
        u_1 = np.cos(100 * dt) * np.heaviside(velocity * dt - grid_x, 1)
    elif droplet:
        centers = [(.1, .5)]
        for center in centers:
            cx = center[0]
            cy = center[1]
            u_0 += np.exp(-100 * ((grid_x - cx) ** 2 + (grid_y - cy) ** 2)) * \
                 np.cos(100 * np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2))
            new_x = grid_x - cx + velocity * dt * (grid_x - cx) / np.sqrt(
                .00001 + (grid_x - cx) ** 2 + (grid_y - cy) ** 2)
            new_y = grid_y - cy + velocity * dt * (grid_y - cy) / np.sqrt(
                .00001 + (grid_x - cx) ** 2 + (grid_y - cy) ** 2)
            u_1 += np.exp(-100 * (new_x ** 2 + new_y ** 2)) * np.cos(100 * np.sqrt(new_x ** 2 + new_y ** 2))
    elif standing_drop:
        cx = .5
        cy = .5
        u_0 = np.heaviside(-np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2), 1)
        new_x = grid_x - cx + velocity * dt * (grid_x - cx) / np.sqrt(.00001 + (grid_x - cx) ** 2 + (grid_y - cy) ** 2)
        new_y = grid_y - cy + velocity * dt * (grid_y - cy) / np.sqrt(.00001 + (grid_x - cx) ** 2 + (grid_y - cy) ** 2)
        u_1 = np.cos(100 * dt) * np.heaviside(-np.sqrt(new_x ** 2 + new_y ** 2), 1)


    # initialize array
    u = np.array([u_0,u_1,u_1])
    for i in range(3):
        u[i, :, :] = x_boundary_conditions(u[i, :, :], xtype='n')
        u[i, :, :] = y_boundary_conditions(u[i, :, :], ytype='n')

    # Main body algorithm
    a = []
    barrier = n_slit_barrier(grid_x, grid_y, 0.5, 1, (0.1, 0.05))
    for t in range(0, 500):
        a.append(np.copy(u[0, :, :]))
        u[2, :, :] = u[1, :, :]
        u[1, :, :] = u[0, :, :]
        u[0, 1:nx - 1, 1:ny - 1] = 2 * u[1, 1:nx - 1, 1:ny - 1] - u[2, 1:nx - 1, 1:ny - 1] + \
                                   (u[1, 2:nx, 1:ny - 1] - 2 * u[1, 1:nx - 1, 1:ny - 1] + u[1, 0:nx - 2, 1:ny - 1] +
                                    u[1, 1:nx - 1, 2:ny] - 2 * u[1, 1:nx - 1, 1:ny - 1] + u[1, 1:nx - 1, 0:ny - 2]) * \
                                   (velocity * dt / dx) ** 2
        u[0, :, :] *= barrier
        u[0, :, :] = x_boundary_conditions(u[0, :, :], psi_prev=u[1, :, :], xtype='a', r=2)
        u[0, :, :] = y_boundary_conditions(u[0, :, :], ytype='n')

        if standing:
            u += np.cos(100 * t * dt) * np.heaviside(-grid_x, 1)
        if standing_drop:
            u += np.cos(100 * t * dt) * np.heaviside(-np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2), 1)

    fig = plt.figure()
    fig.set_dpi(100)
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = fig.add_subplot(1, 1, 1)
    psi = np.ones((nx, ny)) * float('nan')
    meshplot = plt.pcolormesh(grid_x, grid_y, psi)
    plt.clim(-1, 1)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    k = 0

    # Plot the scalar wave
    def animate(i):
        global k
        psi = a[k]
        meshplot.set_array(psi.ravel())
        k += 1


    anim = animation.FuncAnimation(fig, animate, frames=len(a) - 2, interval=20)
    anim.save('2D_wave.gif', fps=30)

    # Plot the power at the far end of the screen
    power_dist, = ax2.plot(y, np.zeros_like(y), color="red", lw=2)
    def animate2(i):
        power_dist.set_ydata(a[i][:, nx - 1] ** 2)


    anim2 = animation.FuncAnimation(fig, animate2, frames=len(a) - 2, interval=20)
    anim2.save('power_dist.gif')

    # Plot the "integral" of the function
    b = []
    for t in range(len(a)):
        b.append(np.sum(a[t]**2))

    plt.plot(np.arange(0,len(a)),b)