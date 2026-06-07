#!/usr/bin/env python3.11
"""
https://www.sfu.ca/~ssurjano/egg.html
"""
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import sin, sqrt, arange, meshgrid, stack, abs
from sys import argv
from gzip import open as gzopen
from math import inf

eggholder = lambda x, k = 47: (-(x[1] + k) * sin(sqrt(abs(x[0]/2 + (x[1]  + k)))) 
                                     -x[0] * sin(sqrt(abs(x[0]   - (x[1]  + k)))))

def main():
    n = 512
    r = range(-n, n+1)
    x = arange(r.start, r.stop)
    xgrid, ygrid = meshgrid(x, x)
    xy = stack([xgrid, ygrid])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    angle=45
    ax.view_init(angle, -angle)
    eggholder_xy=eggholder(xy)
    with gzopen("dataset.txt.gz","wt") as ds:
        ds.write("X1 X2 Y1\n")
        [[ds.write(f"{xy[0][i][j]} {xy[1][i][j]} {eggholder_xy[i][j]}\n") for j in r] for i in r]
    ax.plot_surface(xgrid, ygrid, eggholder_xy, cmap='terrain')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('eggholder(x, y)')
    # set window title only if a canvas manager is available (some backends may not provide one)
    mgr = getattr(fig.canvas, "manager", None)
    if mgr is not None and hasattr(mgr, "set_window_title"):
        try:
            mgr.set_window_title('Eggholder plot')
        except Exception:
            pass
    timeout = inf
    if len(argv) > 2:
        if '-timeout' == argv[1]:
            timeout = int(argv[2]) 
    if not inf == timeout:
        timer = fig.canvas.new_timer(interval=timeout*1000, callbacks=[(plt.close, [], {})])
        timer.start()
    plt.show()
if __name__ == "__main__":
    main()


