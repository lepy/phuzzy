# -*- coding: utf-8 -*-
if __name__ == "__main__":

    import numpy as np
    from phuzzy.contrib.tgo import tgo


    def f(x):
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


    def fv(x):
        return (x[:, 0] ** 2 + x[:, 1] - 11) ** 2 + (x[:, 0] + x[:, 1] ** 2 - 7) ** 2


    bounds = [(-6, 6), (-6, 6)]

    res = tgo(f, bounds, args=(), g_cons=None, g_args=(), n=50)
    print(res)

    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm

    B = 200
    H = 100
    fig = plt.figure(dpi=150, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))
    ax = fig.add_subplot(1, 2, 2)
    ax.set_aspect("equal")
    ax2 = fig.add_subplot(1, 2, 1, projection='3d')
    #     fig.gca(axis=ax2, projection='3d')
    #    ax2 = fig.gca(projection='3d')

    n = 30

    X = np.linspace(-6, 6, n)
    Y = np.linspace(-6, 6, n)
    X, Y = np.meshgrid(X, Y)

    XY = np.vstack((X.ravel(), Y.ravel())).T

    Z = fv(XY)
    Z = Z.reshape(X.shape)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')
    ax2.set_zlabel('$z$')

    ax2.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=1, cmap=cm.viridis,
                     linewidth=0, antialiased=True)

    ax.contour(X, Y, Z, rstride=1, cstride=1, alpha=1, cmap=cm.viridis,
               linewidth=0, antialiased=True, levels=np.linspace(Z.min(), Z.max(), 20))

    #    ax.plot([res.x[0]], [res.x[1]], [f(res.x)], "o")
    for x in res.xl:
        ax.plot([x[0]], [x[1]], "o")

    for x in res.xl:
        ax2.plot([x[0]], [x[1]], [f(x)], "o")
    ax.set_xlim(bounds[0])
    ax.set_xlim(bounds[1])
    fig.tight_layout()
    # fig.savefig("/tmp/tgo_himmelblau.png")
    plt.show()
