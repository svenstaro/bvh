import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import math
from subprocess import Popen, PIPE

# number of benchmarks per power of 10
resolution = 10
# benchmark 1..=10^max rays/triangles
max = 4

#newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
cmap = LinearSegmentedColormap.from_list("speedup", ["red", "white", "green"], N=256)

def plot(colormaps):
    mesh = []
    for x in range(1 * resolution, max * resolution + 1):
        mesh.append(round(math.pow(10, x * (1.0 / resolution))))

    data = np.empty(((max - 1) * resolution + 1, (max - 1) * resolution + 1))

    with np.nditer(data, flags=['multi_index'], op_flags=['writeonly']) as it:
        for x in it:
            rays = mesh[it.multi_index[0]]
            triangles = mesh[it.multi_index[1]]
            samples = 11
            if triangles >= 100 and rays >= 100:
                # these run longer so less intrinsic noise; save time
                # by sampling less
                samples = 3
            proc = Popen([
                "cargo",
                "run",
                "--release",
                "--",
                "--triangles",
                str(triangles),
                "--rays",
                str(rays),
                "--samples",
                str(samples),
            ], stdout=PIPE)
            stdout, _ = proc.communicate()
            x[...] = float(stdout)

    np.random.seed(19680801)
    n = len(colormaps)
    fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
                            layout='constrained', squeeze=False)
    for ax in axs.flat:
        ax.set_title("BVH Speedup Coefficient")
        ax.set_xlabel("Triangles")
        ax.set_ylabel("Rays")
        ax.set_xscale('log')
        ax.set_yscale('log')
        psm = ax.pcolormesh(mesh, mesh, data, cmap=cmap, rasterized=True, vmin=0, vmax=2, shading="gouraud")
        fig.colorbar(psm, ax=ax)
    plt.show()

plot([None])