from matplotlib.colors import LightSource
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# plot_3D: Create a 3D surface plot of a 3D variable over a bathymetric domain.
# ---------------------------------------------------------------


def plot_3D(xr,yr,zr,h,zeta,data,
            name=None,units=None,vmin=None,vmax=None,cmap='RdBu_r',
            angle1=30,angle2=120,norm='linear'):
    """
    Create a 3D surface plot of a 3D variable over a bathymetric domain.

    Parameters
    - xr, yr: 1D arrays of x and y coordinates (domain grid)
    - zr: 3D or 3-level vertical coordinate array (N, ny, nx)
    - h: bathymetry (1D or 2D) used for plotting bathymetry edges
    - zeta: 2D surface elevation used as the top surface z-values
    - data: 3D array of variable values corresponding to zr
    - name, units: optional strings for colorbar label
    - vmin, vmax: optional color limits
    - cmap: colormap name
    - angle1, angle2: view elevation and azimuth for the 3D view
    - norm: 'linear' (default) or 'log' (or a matplotlib.colors.Normalize instance)
    """
    # Prepare grid arrays sized to the vertical levels
    N = zr.shape[0]
    X = np.tile(xr[None, None, :], [N, yr.shape[0], 1])
    Y = np.tile(yr[None, :, None], [N, 1, xr.shape[0]])
    Z = zr

    # Determine sensible vmin/vmax defaults depending on requested norm.
    # For log norm we need positive vmin; for linear we default to symmetric around zero.
    has_positive = np.any(data > 0)
    if isinstance(norm, str) and norm.lower() == 'log':
        if not has_positive:
            raise ValueError("Log norm requested but data contains no positive values.")
        if vmin is None:
            vmin = float(np.nanmin(data[data > 0]))
        if vmax is None:
            vmax = float(np.nanmax(data))
    else:
        if vmin is None:
            vmin = -np.nanmax(np.abs(data))
        if vmax is None:
            vmax = -vmin

    # Build a matplotlib Normalize object based on the 'norm' parameter.
    if isinstance(norm, str):
        if norm.lower() in ('linear', 'lin'):
            norm_obj = mcolors.Normalize(vmin=vmin, vmax=vmax)
        elif norm.lower() in ('log', 'lognorm'):
            # ensure strictly positive vmin for log
            eps = max(1e-12, np.nanmin(data[data > 0])) if has_positive else 1e-12
            vmin_log = max(eps, vmin)
            if vmin_log <= 0:
                vmin_log = eps
            norm_obj = mcolors.LogNorm(vmin=vmin_log, vmax=vmax)
        else:
            raise ValueError("Unsupported norm string. Use 'linear' or 'log', or pass a Normalize instance.")
    elif isinstance(norm, mcolors.Normalize):
        # User provided a Normalize instance directly
        norm_obj = norm
    else:
        # Attempt to treat the provided object as a Normalize-like object
        try:
            test = norm.vmin  # noqa: F841
            norm_obj = norm
        except Exception:
            raise ValueError("norm must be 'linear', 'log', a matplotlib.colors.Normalize instance, or similar.")

    # Light source for shaded relief / hillshade effect
    ls = LightSource(azdeg=315, altdeg=45)
    vert_exag = 1.0  # vertical exaggeration for relief visualization

    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # Top surface (use last vertical level as the top layer)
    X_top = X[-1, :, :]
    Y_top = Y[-1, :, :]
    Z_top = zeta
    data_top = data[-1, :, :]

    # Arguments passed to LightSource.shade to produce an RGB image
    shade_args= {'cmap': plt.get_cmap(cmap),
                 'vert_exag': vert_exag,
                 'blend_mode': 'soft',
                 'norm': norm_obj}
    surface_args = { 'rstride': 1, 'cstride': 1,
                     'linewidth': 0, 'antialiased': False, 'shade': False}

    # For log normalization, ensure strictly positive values for shading
    if isinstance(norm_obj, mcolors.LogNorm):
        eps = max(1e-12, np.nanmin(data[data > 0]))  # safe positive minimum
        data_top_plot = np.clip(data_top, eps, None)
    else:
        data_top_plot = data_top

    # Create shaded RGB surface for the top
    rgb_top = ls.shade(data_top_plot, **shade_args)

    ax.plot_surface(X_top, Y_top, Z_top, facecolors=rgb_top,
                    **surface_args)

    # Side surface at y = Y.max() (one lateral wall)
    X_side = X[:, -1, :]
    Z_side = Z[:, -1, :]
    Y_side_const = np.ones_like(X_side) * Y.max()
    data_side = data[:, -10, :]

    if isinstance(norm_obj, mcolors.LogNorm):
        data_side_plot = np.clip(data_side, eps, None)
    else:
        data_side_plot = data_side

    rgb_side = ls.shade(data_side_plot, **shade_args)
    ax.plot_surface(X_side, Y_side_const, Z_side, facecolors=rgb_side,
                    **surface_args)

    # Side surface at y = Y.min() (opposite lateral wall)
    Y_side_const2 = np.ones_like(X_side) * Y.min()
    data_side2 = data[:, 0, :]
    Z_side2 = Z[:, 0, :]
    if isinstance(norm_obj, mcolors.LogNorm):
        data_side2_plot = np.clip(data_side2, eps, None)
    else:
        data_side2_plot = data_side2
    rgb_side2 = ls.shade(data_side2_plot, **shade_args)
    ax.plot_surface(X[:, 0, :], Y_side_const2, Z_side2, facecolors=rgb_side2,
                    **surface_args)

    # Side surface at x = X.min() (front/back wall)
    X_side_const2 = np.ones_like(X[:, :, 0]) * X.min()
    Y_side2 = Y[:, :, 0]
    Z_side2 = Z[:, :, 0]
    data_side3 = data[:, :, 0]
    if isinstance(norm_obj, mcolors.LogNorm):
        data_side3_plot = np.clip(data_side3, eps, None)
    else:
        data_side3_plot = data_side3

    rgb_side3 = ls.shade(data_side3_plot, **shade_args)
    ax.plot_surface(X_side_const2, Y_side2, Z_side2, facecolors=rgb_side3,
                    **surface_args)
                    
    # Optional: draw bathymetry edges along the domain for context
    ax.plot(X[0, 0, :], Y.max(), -h[0, :], 'k', linewidth=0.8)
    ax.plot(X[0, 0, :], Y.min(), -h[0, :], 'k', linewidth=0.8)

    # Axes labels and aspect ratio
    ax.set(xlabel='X [m]', ylabel='Y [m]', zlabel='Z [m]')
    ax.view_init(angle1,angle2)
    ax.set_box_aspect([1., 1., 0.4], zoom=0.9)

    # Set explicit axis limits to the domain extents
    x_min, x_max = float(xr.min()), float(xr.max())
    y_min, y_max = float(yr.min()), float(yr.max())
    z_min, z_max = float(np.nanmin(Z)), float(np.nanmax(Z))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Make panes transparent and set their edge color to white for a clean look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Colorbar: create a mappable with the same cmap and normalization
    mappable = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=norm_obj)
    mappable.set_array([])  # required for colorbar
    fig.colorbar(mappable, ax=ax, fraction=0.02, pad=0.1, label=f'{name} [{units}]')
    plt.tight_layout()

    plt.show()

