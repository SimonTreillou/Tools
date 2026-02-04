from matplotlib.colors import LightSource
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# isosurface_vertices: Compute isosurface vertices and faces using marching cubes.
# plot_CIbar_loglog: Plot a confidence interval bar on a log-log power spectrum plot.
# plot_3D: Create a 3D surface plot of a 3D variable over a bathymetric domain.
# plot_bar_simple: Simple, robust bar plot with sensible defaults and comments.
# plot_isosurface: Compute & plot an isosurface of Qx using vertical interpolation, marching cubes, and matplotlib.
# plot_stack_fraction: Stacked bar plot of fractional contributions.
# ---------------------------------------------------------------

def isosurface_vertices(volume, level, spacing):
    """
    Compute vertices and faces of an isosurface using scikit-image's marching cubes.

    Parameters
    - volume: 3D numpy array
    - level: isosurface value
    - spacing: voxel spacing as (sx, sy, sz)

    Returns
    - verts, faces: arrays suitable for 3D surface rendering
    """
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError("volume must be a 3D array")

    # Normalize spacing to a 3-tuple
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)
    elif np.isscalar(spacing):
        spacing = (float(spacing),) * 3
    else:
        spacing = tuple(float(s) for s in spacing)

    verts, faces, _, _ = marching_cubes(volume=volume, level=level, spacing=spacing)
    return verts, faces

def plot_CIbar_loglog(f, S, ci_upper, color='k', x_pos=None, y_pos=None, lw=3):
    """
    Plot a confidence interval (CI) bar on a log-log power spectrum plot.

    Parameters
    ----------
    f : array-like
        Frequency array.
    S : array-like
        Power spectral density array.
    ci_upper : float
        Upper bound multiplier for the confidence interval (from welch_spectrum_CI).
    color : str, optional
        Color of the CI bar (default: 'k' for black).
    x_pos : float, optional
        X position (frequency) for the CI bar. Defaults to 80% of max frequency.
    y_pos : float, optional
        Y position (PSD) for the CI bar. Defaults to 80% of max PSD value.
    lw : int, optional
        Line width of the CI bar (default: 3).

    Notes
    -----
    The CI bar is drawn vertically at x_pos, centered at y_pos, with height determined by ci_upper.
    Intended for use on log-log plots.
    """
    if x_pos is None:
        x_pos = f[-1] * 0.8  # near top-right, 80% of max freq
    if y_pos is None:
        y_center = max(S) * 0.8  # position around the top of PSD curve
    else:
        y_center = y_pos
    # CI width (half height in log10 units)
    ci_half = ci_upper * S[0]  # same everywhere

    # Draw the bar (vertical line)
    plt.vlines(x=x_pos, ymin=y_center - ci_half, ymax=y_center + ci_half,
               colors=color, linewidth=lw)

    # Add small horizontal “caps”
    plt.hlines([y_center - ci_half, y_center + ci_half],
               x_pos * 0.95, x_pos * 1.05, colors=color, linewidth=lw)
    
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

def plot_bar_simple(names, values, title="", ylabel="", vmin=None, vmax=None,
                    color="#1f77b4", neg_color="#d62728", annotate=False,
                    width=0.6, figsize=(10, 5), dpi=300, rotate=45):
    """
    Simple, robust bar plot with sensible defaults and comments.

    Parameters
    - names: sequence of labels (will be converted to strings)
    - values: sequence/array of numeric values (can contain NaN)
    - title: plot title (string)
    - ylabel: y-axis label (string)
    - vmin, vmax: explicit y-limits. If None they are chosen intelligently:
        * if all values >= 0: vmin = 0, vmax = max(values)*1.1
        * if all values <= 0: vmax = 0, vmin = min(values)*1.1
        * otherwise symmetric around zero with 10% margin
    - color: color for non-negative bars (string or list)
    - neg_color: color for negative bars when mixed-sign data is present
    - annotate: if True, numeric values are drawn above/below each bar
    - width: bar width (0..1)
    - figsize, dpi: figure size and resolution
    - rotate: x-tick label rotation in degrees

    Returns
    - fig, ax: matplotlib figure and axes for further customization
    """
    # Convert inputs to numpy arrays for safe numeric ops
    values = np.asarray(values, dtype=float)
    names = [str(n) for n in names]

    if values.shape[0] != len(names):
        raise ValueError("Length of 'names' and 'values' must match.")

    # Compute sensible vmin/vmax defaults if not provided
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        # fallback if all are NaN or infinite
        vmin_default, vmax_default = -1.0, 1.0
    else:
        vmin_data, vmax_data = float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals))
        if vmin is None and vmax is None:
            if vmin_data >= 0:
                vmin_default = 0.0
                vmax_default = vmax_data * 1.1 if vmax_data != 0 else 1.0
            elif vmax_data <= 0:
                vmax_default = 0.0
                vmin_default = vmin_data * 1.1 if vmin_data != 0 else -1.0
            else:
                m = max(abs(vmin_data), abs(vmax_data)) * 1.1
                vmin_default, vmax_default = -m, m
        elif vmin is None:
            vmin_default = vmin_data if np.isfinite(vmin_data) else (vmax * -1.0)
            vmin_default = min(vmin_default, vmax)  # be conservative
            vmax_default = vmax
        elif vmax is None:
            vmax_default = vmax_data if np.isfinite(vmax_data) else (vmin * -1.0)
            vmax_default = max(vmax_default, vmin)  # be conservative
            vmin_default = vmin
        else:
            vmin_default, vmax_default = vmin, vmax

    # Create the figure and axis
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)

    # Choose bar colors:
    # - If user passed a single color string and data contains mixed signs, use pos/neg colors.
    # - If user passed a sequence of colors of matching length, use it directly.
    if hasattr(color, "__iter__") and not isinstance(color, str):
        if len(color) != len(values):
            raise ValueError("If 'color' is a sequence, it must match length of 'values'.")
        colors = list(color)
    else:
        if np.any(values < 0) and np.any(values > 0):
            # mixed signs -> use two-tone coloring
            colors = [color if v >= 0 or np.isnan(v) else neg_color for v in values]
        else:
            colors = color  # single color for all bars

    # Plot bars (Matplotlib will skip NaN values)
    bars = ax.bar(names, values, color=colors, edgecolor='black',
                  linewidth=0.5, width=width)

    # Draw horizontal zero line for reference
    ax.axhline(0, color='0.2', linewidth=0.8, linestyle='--', zorder=0)

    # Set axis labels, limits and appearance
    ax.set_ylabel(ylabel)
    ax.set_ylim([vmin_default, vmax_default])
    ax.set_title(title, loc='right')
    plt.xticks(rotation=rotate, ha='right')

    # Remove top/right spines for a cleaner look
    if hasattr(ax, "spines"):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    ax.grid(axis='y', alpha=0.3)

    # Optional: annotate each bar with its numeric value
    if annotate:
        for bar, val in zip(bars, values):
            if np.isnan(val):
                txt = "NaN"
            else:
                txt = f"{val:.3g}"
            height = bar.get_height()
            # Place annotation above positive bars, below negative bars
            if np.isnan(height):
                y = 0
                va = 'bottom'
            elif height >= 0:
                y = height + (vmax_default - vmin_default) * 0.01
                va = 'bottom'
            else:
                y = height - (vmax_default - vmin_default) * 0.01
                va = 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, y, txt,
                    ha='center', va=va, fontsize=8)

    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_isosurface(Qx, xr, yr, zr, iso_value,
                          h=None,                  # (Ny, Nx) bathy depth (positive down)
                          plot_bathy=True,
                          bathy_stride=4,
                          Nz_out=None, zmin=None, zmax=None,
                          xmin=None, xmax=None, ymin=None, ymax=None,
                          z_positive_up=True,
                          base_color=[189, 20, 8],
                          elev=30, azim=130):
    """
    Compute & plot an isosurface of Qx using:
      - vertical interpolation from terrain-following zr -> regular z-levels
      - marching cubes
      - matplotlib Poly3DCollection
      + optional bathymetry surface

    Qx: (Nz, Ny, Nx)
    zr: (Nz, Ny, Nx) physical z (same order as Qx)
    xr: (Nx,), yr: (Ny,)
    h : (Ny, Nx) bathymetry depth (positive down) [optional]
    """

    Nz, Ny, Nx = Qx.shape
    dx = float(xr[1] - xr[0])
    dy = float(yr[1] - yr[0])

    # --- 1) Build regular z-levels (Cartesian) ---
    if zmin is None: zmin = np.nanmin(zr)
    if zmax is None: zmax = np.nanmax(zr)
    if xmin is None: xmin = np.nanmin(xr)
    if xmax is None: xmax = np.nanmax(xr)
    if ymin is None: ymin = np.nanmin(yr)
    if ymax is None: ymax = np.nanmax(yr)
    if Nz_out is None: Nz_out = Nz * 2  # finer z-grid helps marching cubes

    z_levels = np.linspace(zmin, zmax, Nz_out)  # ascending

    # --- 2) Interpolate Qx(zr) -> Qx(z_levels) column-by-column ---
    Qreg = np.full((Nz_out, Ny, Nx), np.nan, dtype=float)

    for j in range(Ny):
        for i in range(Nx):
            zcol = zr[:, j, i]
            qcol = Qx[:, j, i]

            m = np.isfinite(zcol) & np.isfinite(qcol)
            if m.sum() < 2:
                continue

            zc = zcol[m]
            qc = qcol[m]

            idx = np.argsort(zc)
            zc = zc[idx]
            qc = qc[idx]

            Qreg[:, j, i] = np.interp(z_levels, zc, qc, left=np.nan, right=np.nan)

    # marching_cubes needs finite values: set NaNs to a value far away
    Qwork = Qreg.copy()
    nan_mask = ~np.isfinite(Qwork)
    if np.any(~nan_mask):
        finite_min = np.nanmin(Qwork)
        Qwork[nan_mask] = finite_min - 1e6
    else:
        raise ValueError("All values are NaN after interpolation.")

    # --- 3) Marching cubes on a REGULAR grid ---
    dz = float(z_levels[1] - z_levels[0])
    verts, faces, normals, values = marching_cubes(
        Qwork, level=iso_value, spacing=(dz, dy, dx)
    )
    # Compute per-face normals (average of the 3 vertex normals)
    face_normals = normals[faces].mean(axis=1)

    # Convert verts to physical coordinates
    x0 = float(xr[0])
    y0 = float(yr[0])
    z0 = float(z_levels[0])

    Z = verts[:, 0] + z0
    Y = verts[:, 1] + y0
    X = verts[:, 2] + x0

    XYZ = np.c_[X, Y, Z]
    tri = np.stack([XYZ[faces[:, 0]], XYZ[faces[:, 1]], XYZ[faces[:, 2]]], axis=1)

    # --- 4) Plot with matplotlib ---
    fig = plt.figure(figsize=(9, 7),dpi=300)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    # --- Shading for isosurface (rollers) ---
    ls = LightSource(azdeg=90, altdeg=20)

    # Light direction (unit vector)
    light_dir = ls.direction  # shape (3,)

    # Normalize face normals
    fn = face_normals.copy()
    fn /= np.linalg.norm(fn, axis=1)[:, None]

    # Lambertian shading: dot(n, light_dir)
    intensity = np.dot(fn, light_dir)
    intensity = np.clip(intensity, 0.3, 1.0)

    # Map to grayscale or a colormap
    # Here: bluish rollers
    base_color = np.array(base_color)/255  # RGB
    facecolors = intensity[:, None] * base_color[None, :]

    mesh = Poly3DCollection(tri, linewidths=0.0, alpha=0.9)
    mesh.set_facecolor(facecolors)     # <-- shaded rollers
    mesh.set_edgecolor('none')
    mesh.set_linewidth(0.0)
    mesh.set_zorder(20)
    mesh.set_antialiased(False) 
    ax.add_collection3d(mesh)

    # --- 5) Bathymetry overlay ---
    if plot_bathy and (h is not None):
        if h.shape != (Ny, Nx):
            raise ValueError(f"h must have shape (Ny,Nx)={(Ny,Nx)}, got {h.shape}")

        X2, Y2 = np.meshgrid(xr, yr)  # (Ny,Nx)

        # If z is positive upward, bottom is negative (zb=-h). If z positive downward, zb=+h.
        zb = (-h) if z_positive_up else (h)

        bathy = ax.plot_surface(
            X2[::bathy_stride, ::bathy_stride],
            Y2[::bathy_stride, ::bathy_stride],
            zb[::bathy_stride, ::bathy_stride],
            rstride=30, cstride=30, linewidth=0, alpha=0.8,
            color=np.array([235, 143, 5])/255, antialiased=True, shade=True
        )
        bathy.set_zorder(0)

    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.set_xlabel("x [m]",fontsize=11,rotation=0)
    ax.set_ylabel("y [m]",fontsize=11)
    ax.xaxis.set_label_coords(0.5, -1.33)
    ax.yaxis.set_label_coords(1.05, 0.5)
    ax.set_zlabel("")
    ax.text2D(0.98, 0.67, "Depth [m]", transform=ax.transAxes,
            rotation=0, va="center", ha="left", fontsize=11)
    # Improve spacing so it is not clipped
    ax.tick_params(axis='z', pad=5)
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))

    # Set limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # Make sure zlim includes bathy if present
    zlo = z_levels.min()
    zhi = z_levels.max()
    if plot_bathy and (h is not None):
        zb_min = np.nanmin(((-h) if z_positive_up else (h)))
        zb_max = np.nanmax(((-h) if z_positive_up else (h)))
        zlo = min(zlo, zb_min)
        zhi = max(zhi, zb_max)
    ax.set_zlim(zmin, zmax)
    
    # Set view angle
    ax.view_init(elev=elev,azim=azim) #30,130
    
    # Set aspect ratio
    ax.set_box_aspect((90/110, 1.0, 0.2))
    fig.tight_layout()
    return fig, ax, (verts, faces), (X, Y, Z), z_levels, Qreg


def plot_stack_fraction(data, names, t=None, figsize=(12,6), cmap='tab10',
                        normalize=True, smooth=None, alpha=0.95,
                        legend_cols=2, ylabel="Fraction of total |budget|"):
    """
    Stacked-area plot of fractional contributions.

    Parameters
    - data: 2D array (nterms, nt) of contributions (can be signed)
    - names: sequence of nterms labels
    - t: 1D time axis (length nt). If None uses np.arange(nt)
    - figsize: figure size tuple passed to plt.subplots
    - cmap: matplotlib colormap name for colors
    - normalize: if True convert to absolute contributions and normalize at each time to sum=1
    - smooth: int window for simple moving-average smoothing (applied to fractions)
    - alpha: transparency for stacked areas
    - legend_cols: number of columns for the legend
    - ylabel: y-axis label

    Returns
    - fig, ax: matplotlib figure and axes for further customization
    """
    import matplotlib.pyplot as plt

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be 2D with shape (nterms, nt)")

    nterms, nt = data.shape
    if t is None:
        t = np.arange(nt)
    else:
        t = np.asarray(t)
        if t.shape[0] != nt:
            raise ValueError("time axis length must match data second dimension")

    # Work with absolute contributions for fraction-of-budget plots
    frac = np.abs(data)
    # Replace NaNs with 0 for plotting / normalization
    frac = np.where(np.isfinite(frac), frac, 0.0)

    # Normalize to sum=1 at each time
    if normalize:
        tot = np.sum(frac, axis=0)
        # avoid division by zero
        zero_mask = tot == 0
        tot[zero_mask] = 1.0
        frac = frac / tot

    # Optional smoothing (moving average) along time axis
    if smooth is not None and smooth is not False:
        w = int(smooth)
        if w > 1:
            kernel = np.ones(w) / w
            frac_smooth = np.empty_like(frac)
            for i in range(nterms):
                frac_smooth[i, :] = np.convolve(frac[i, :], kernel, mode='same')
            frac = frac_smooth

    # Order terms so largest mean contribution is at bottom of stack for readability
    mean_frac = np.mean(frac, axis=1)
    order = np.argsort(mean_frac)[::-1]  # descending
    frac_ord = frac[order, :]
    names_ord = [names[i] for i in order]
    mean_frac_ord = mean_frac[order]

    # Colors
    cmap_obj = plt.get_cmap(cmap)
    # get distinct colors (wrap if nterms > cmap.N)
    colors = [cmap_obj(i % getattr(cmap_obj, "N", 10)) for i in range(nterms)]

    # Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.stackplot(t, frac_ord, labels=[f"{n} ({100*mf:.1f}%)" for n, mf in zip(names_ord, mean_frac_ord)],
                 colors=colors[:nterms], alpha=alpha)
    ax.set_ylim(0, 1 if normalize else None)
    ax.set_xlim(t.min(), t.max())
    ax.set_xlabel("Time index")
    ax.set_ylabel(ylabel)
    ax.set_title("Normalized fractional contributions to the vorticity budget", loc='left')
    ax.grid(alpha=0.25)

    # Legend: put to the right if many terms, show mean % in label
    ax.legend(ncol=legend_cols, fontsize=9, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)

    plt.tight_layout(rect=(0, 0, 0.85, 1.0))  # leave room for legend

    return fig, ax