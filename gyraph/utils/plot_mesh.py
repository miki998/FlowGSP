"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""


from typing import Optional
from .utils import np
from .plot_utils import plt, matplotlib
from mpl_toolkits.mplot3d import art3d
import matplotlib.tri as mtri
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter


def plot_signal_on_regular_surface(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    grid_size: int,
    upsample: dict,
    sigs: np.ndarray,
    surface_type: str,
    curve_scale: float,
    parabola_scale: float,
    bump_scale: float = 30.0,
    labels: list = None,
    elev: int = 56,
    azim: int = 24,
    smooth_sigma: Optional[float] = None,
) -> matplotlib.figure.Figure:
    fig = plt.figure(figsize=(2.5 * len(sigs), 2))

    for i, sig in enumerate(sigs):
        ax = fig.add_subplot(1, len(sigs), i + 1, projection="3d")

        # --- Smooth the signal surface by interpolating to a finer grid ---
        S = sig.reshape(grid_size, grid_size)
        # Optional in-grid smoothing for the signal surface
        if smooth_sigma is not None and smooth_sigma > 0:
            try:
                S = gaussian_filter(S, sigma=smooth_sigma, mode="nearest")
            except Exception:
                pass

        cmap = plt.get_cmap("RdBu_r")
        # Other nice diverging colormaps to try: "RdBu", "PRGn", "BrBG", "PuOr"
        symmetric_vminmax = max(abs(np.nanmin(S)), abs(np.nanmax(S)))
        norm = plt.Normalize(vmin=-symmetric_vminmax, vmax=symmetric_vminmax)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])  # needed for older Matplotlib versions

        if upsample > 1:
            x1d = X[0, :]  # original x-grid
            y1d = Y[:, 0]  # original y-grid
            xi = np.linspace(x1d.min(), x1d.max(), grid_size * upsample)
            yi = np.linspace(y1d.min(), y1d.max(), grid_size * upsample)
            Xf, Yf = np.meshgrid(xi, yi)

            try:
                # Small positive 's' gives slight smoothing; 0.0 is pure interpolation
                smooth_s = 0.1
                spline = RectBivariateSpline(y1d, x1d, S, kx=3, ky=3, s=smooth_s)
                Sf = spline(yi, xi)
            except Exception:
                # Fallback: bilinear-like upsampling (no extra dependency)
                Sf = np.repeat(np.repeat(S, upsample, axis=0), upsample, axis=1)

            # Recompute surface z on the finer grid (keeps exact geometry)
            if surface_type == "inverted_parabola":
                Zf = (
                    -((curve_scale * Xf) ** 2 + (curve_scale * Yf) ** 2)
                    / parabola_scale
                )
            elif surface_type == "hyperbolic_paraboloid":
                Zf = (
                    (curve_scale * Xf) ** 2 - (curve_scale * Yf) ** 2
                ) / parabola_scale
            elif surface_type == "two_hole_curve":
                Zf = (
                    np.sin(np.pi * Xf / Xf.max())
                    * np.sin(np.pi * Yf / Yf.max())
                    / parabola_scale
                )

            Zf += bump_scale * Sf  # Add the signal to increase height

            facecolors_f = cmap(norm(Sf)).reshape(-1, 4)

            tri = mtri.Triangulation(Xf.flatten(), Yf.flatten())
            tri_facescolors_f = facecolors_f[tri.triangles].mean(axis=1)
            surf = ax.plot_trisurf(
                Xf.flatten(),
                Yf.flatten(),
                Zf.flatten(),
                triangles=tri.triangles,
                linewidth=0.1,
                antialiased=True,
                shade=False,
                alpha=1,
                edgecolor="black",
            )
            surf.set_facecolors(tri_facescolors_f)
        else:
            # Color the *surface* by the graph signal (reshape signal back to grid)
            facecolors = cmap(norm(S)).reshape(-1, 4)

            tri = mtri.Triangulation(X.flatten(), Y.flatten())
            tri_facescolors = facecolors[tri.triangles].mean(axis=1)
            surf = ax.plot_trisurf(
                X.flatten(),
                Y.flatten(),
                Z.flatten(),
                triangles=tri.triangles,
                linewidth=0.1,
                antialiased=True,
                shade=False,
                alpha=1,
                edgecolor="black",
            )
            surf.set_facecolors(tri_facescolors)

        cbar = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=-0.06, shrink=0.45)
        if i == len(sigs) - 1:
            cbar.set_label("Signal value")

        ax.set_xlabel("X", fontsize=4.5, labelpad=2)
        ax.set_ylabel("Y", fontsize=4.5, labelpad=2)
        ax.set_zlabel("Z", fontsize=4.5, labelpad=2)

        if labels is not None:
            ax.set_title(labels[i], fontsize=6)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_zlim(1.5 * Z.min(), 1.5 * Z.max())

        # # Reserve a fixed strip on the right for the colorbar and re-center the axes.
        # ax.set_position([0.06, 0.06, 0.80, 0.88])  # [left, bottom, width, height]
        # cbar.ax.set_position([0.85, 0.25, 0.03, 0.50])

    plt.show()
    return fig


def signal2face(
    signal: np.ndarray,
    faces_idx: np.ndarray,
    region_idx: np.ndarray,
    vlabels: np.ndarray,
) -> np.ndarray:
    """
    Convert signal on vertices to signal on triangulated faces.

    Parameters:
    -----------
        signal (np.ndarray): Signal values on vertices.
        faces_idx (np.ndarray): Indices of vertices that form each face.
        region_idx (np.ndarray): Region indices for each vertex.
        vlabels (np.ndarray): Labels for each vertex.

    Returns:
    --------
        faces_signal (np.ndarray): Signal values on each face.
    """

    d = {r: ridx for ridx, r in enumerate(region_idx)}
    d[0] = -1  # background value
    vertex_signal = signal[np.array([d[k] for k in vlabels])]
    faces_signal = vertex_signal[faces_idx].mean(axis=1)
    return faces_signal


def plot_mesh(
    vertices: np.ndarray,
    f_index: np.ndarray,
    f_colors: np.ndarray,
    title: str = "The Macaque Brain",
    view_init: tuple = (0, -136),
    eps: float = 0.1,
    ax: matplotlib.axes = None,
    fig: matplotlib.figure.Figure = None,
    cmap: str = "viridis",
):
    """
    Plot a 3D mesh given the vertices, face indices, and face colors.

    Parameters:
    -----------
        vertices (np.ndarray): Array of vertex coordinates.
        f_index (np.ndarray): Array of face indices.
        f_colors (np.ndarray): Array of face colors.
        title (str, optional): Title of the plot, defaults to "The Macaque Brain".
        view_init (tuple, optional): Initial view angles for the 3D plot, defaults to (0, -136).
        eps (float, optional): Epsilon value for setting plot limits, defaults to 0.1.
        ax (matplotlib.axes, optional): Existing 3D axes to plot on.
        fig (matplotlib.figure.Figure, optional): Existing figure to plot on.

    Returns:
    --------
        None
    """

    if ax is None or fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    norm = plt.Normalize(f_colors.min(), f_colors.max())
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(norm(f_colors))

    pc = art3d.Poly3DCollection(vertices[f_index], facecolors=colors)

    ax.add_collection(pc)
    ax.set_xlim(
        vertices[:, 0].min() + vertices[:, 0].min() * eps,
        vertices[:, 0].max() + vertices[:, 0].max() * eps,
    )
    ax.set_ylim(
        vertices[:, 1].min() + vertices[:, 1].min() * eps,
        vertices[:, 1].max() + vertices[:, 1].max() * eps,
    )
    ax.set_zlim(
        vertices[:, 2].min() + vertices[:, 2].min() * eps,
        vertices[:, 2].max() + vertices[:, 2].max() * eps,
    )

    ax.set_title(title)
    ax.axis("off")

    ax.view_init(view_init[0], view_init[1])
    if ax is None or fig is None:
        plt.show()
