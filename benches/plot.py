import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import math
from subprocess import Popen, PIPE

# number of benchmarks per power of 10
resolution = 4
# benchmark 1..=10^max rays/triangles
max = 4

all_benchmarks = [
    "traverse",
    "traverse_iterator",
    "nearest_traverse_iterator",
    "nearest_child_traverse_iterator"
]

cmap = LinearSegmentedColormap.from_list("speedup", ["red", "white", "green"], N=256)

def bench(rays, triangles, samples, benchmark):
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
        "--benchmark",
        benchmark
    ], stdout=PIPE)
    stdout, _ = proc.communicate()
    return float(stdout)

def plot(colormaps):
    mesh = []
    for x in range(0, max * resolution + 1):
        mesh.append(round(math.pow(10, x * (1.0 / resolution))))

    n = len(all_benchmarks)
    fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
                            layout='constrained', squeeze=False)
    
    for (ax, benchmark) in zip(axs.flat, all_benchmarks):
        data = np.empty((max * resolution + 1, max * resolution + 1))

        with np.nditer(data, flags=['multi_index'], op_flags=['writeonly']) as it:
            for x in it:
                rays = mesh[it.multi_index[0]]
                triangles = mesh[it.multi_index[1]]
                samples = 51
                if triangles < 50:
                    # sample more due to counteract observed noise
                    samples = 501
                elif triangles >= 100 and rays >= 100:
                    # these run longer so less intrinsic noise; save time
                    # by sampling less
                    samples = 3
                x[...] = bench(rays, triangles, samples, benchmark)

        ax.set_title(benchmark + "\nBVH Speedup Coefficient")
        ax.set_xlabel("Triangles")
        ax.set_ylabel("Rays")
        ax.set_xscale('log')
        ax.set_yscale('log')
        psm = ax.pcolormesh(mesh, mesh, data, cmap=cmap, rasterized=True, vmin=0, vmax=2, shading="gouraud")
        fig.colorbar(psm, ax=ax)

    plt.show()

plot([None])
