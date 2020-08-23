import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_pointcloud(x, y, z, show=True, show_axis=True, marker='.', s=1, alpha=1.0, figsize=(5, 5), elev=90,
                    azim=0, axis=None, title=None, *args, **kwargs):
    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = Axes3D(fig)
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    ax.scatter3D(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if not show_axis:
        plt.axis('off')

    if show:
        plt.show()

    return fig


def bold_string(x):
    start = "\033[1m"
    end = "\033[0;0m"
    return start + x + end