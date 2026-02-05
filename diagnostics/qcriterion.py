import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from matplotlib.colors import LightSource

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# gradients_3d: Compute spatial gradients of a 3D field
# compute_Q_components: Compute Q-criterion and its components from velocity fields
# normalize_volume: Normalize a 3D volume to its maximum absolute value
# isosurface_vertices: Compute isosurface vertices and faces using marching cubes
# plot_isosurface: Compute and plot an isosurface of Qx with optional bathymetry
# plot_two_isosurfaces: Plot two isosurfaces (Qa and Qb) on the same domain with bathymetry
# ---------------------------------------------------------------

def gradients_3d(field):
    """
    Calculate the gradients of a 3D field along all three axes.

    This function computes the spatial gradients of a 3D scalar field using
    numpy's gradient function, which applies centered differences for interior
    points and forward/backward differences at boundaries.

    Parameters
    ----------
    field : ndarray
        A 3D array with shape (N, M, L) representing the scalar field,
        where dimensions correspond to (z, y, x) axes respectively.

    Returns
    -------
    dfdy : ndarray
        Gradient of the field along the y-axis, same shape as input.
    dfdy : ndarray
        Gradient of the field along the z-axis, same shape as input.
    dfdx : ndarray
        Gradient of the field along the x-axis, same shape as input.

    Notes
    -----
    The function returns gradients in (y, z, x) order to match the naming
    convention used elsewhere in the codebase, even though np.gradient
    returns them in (z, y, x) order based on the input field shape.
    """
    # Returns gradients along (z,y,x) axes
    # np.gradient assumes axis order; here field shape is (N, M, L)
    dfdz, dfdy, dfdx = np.gradient(field)
    return dfdy, dfdz, dfdx  # match (y,z,x) naming used later


def compute_Q_components(ut, vt, wt, dx, dy, dz3):
    """
    Compute Q-criterion components for vortex identification in 3D flow fields.
    
    The Q-criterion is a scalar field used to identify vortex cores in fluid dynamics.
    This function calculates the total Q value and its directional components (Qx, Qy, Qz)
    based on velocity field gradients.
    
    Parameters
    ----------
    ut : ndarray
        3D array of u-component velocities with shape (N, M, L).
    vt : ndarray
        3D array of v-component velocities with shape (N, M, L).
    wt : ndarray
        3D array of w-component velocities with shape (N, M, L).
    dx : float
        Grid spacing in x-direction.
    dy : float
        Grid spacing in y-direction.
    dz3 : float
        Grid spacing in z-direction.
    
    Returns
    -------
    Q : ndarray
        Total Q-criterion scalar field with shape (N, M, L).
        Positive values indicate vortex cores.
    Qx : ndarray
        X-component contribution to Q-criterion with shape (N, M, L).
    Qy : ndarray
        Y-component contribution to Q-criterion with shape (N, M, L).
    Qz : ndarray
        Z-component contribution to Q-criterion with shape (N, M, L).
    
    Notes
    -----
    The Q-criterion is defined as Q = 0.5 * (Ω² - S²), where Ω is the vorticity
    tensor and S is the strain-rate tensor. Positive Q values indicate rotation
    dominance and potential vortex regions.
    """
    # ut, vt, wt: (N, M, L)
    duy, duz, dux = gradients_3d(ut)
    dvy, dvz, dvx = gradients_3d(vt)
    dwy, dwz, dwx = gradients_3d(wt)

    # Scale spatial derivatives
    dux = dux / dx
    dvy = dvy / dy
    dwz = dwz / dz3
    duy = duy / dy
    dvx = dvx / dx
    duz = duz / dz3
    dvz = dvz / dz3
    dwy = dwy / dy
    dwx = dwx / dx

    Q = -0.5 * (
        (dux ** 2) + (dvy ** 2) + (dwz ** 2)
        + 2 * duy * dvx
        + 2 * duz * dwx
        + 2 * dvz * dwy
    )
    Qx = -duz * dwx - 0.5 * (dux ** 2)
    Qy = -dvz * dwy - 0.5 * (dvy ** 2)
    Qz = -duy * dvx - 0.5 * (dwz ** 2)

    return Q, Qx, Qy, Qz


def normalize_volume(vol):
    """
    Normalize a volume array by dividing by its maximum absolute value.

    Parameters
    ----------
    vol : np.ndarray
        Input volume array to be normalized.

    Returns
    -------
    np.ndarray
        Normalized volume array divided by its maximum absolute value.
        If the maximum absolute value is None or 0, returns the original array unchanged.

    Examples
    --------
    >>> vol = np.array([1, 2, 3, 4, 5])
    >>> normalized = normalize_volume(vol)
    >>> np.allclose(normalized, [0.2, 0.4, 0.6, 0.8, 1.0])
    True
    """
    m = np.nanmax(np.abs(vol))
    if m is None or m == 0:
        return vol
    return vol / m

def isosurface_vertices(volume, level, spacing):
    """
    Extract isosurface vertices and faces from a 3D volume using the marching cubes algorithm.
    
    Parameters
    ----------
    volume : ndarray
        A 3D array representing the volumetric data.
    level : float
        The isosurface level value at which to extract the surface.
    spacing : tuple of float
        The physical spacing of voxels in each dimension (dx, dy, dz).
    
    Returns
    -------
    verts : ndarray
        An (N, 3) array of vertex coordinates on the isosurface.
    faces : ndarray
        An (M, 3) array of triangular face indices referencing vertices.
    
    Notes
    -----
    This function wraps scikit-image's marching_cubes algorithm to generate
    an isosurface mesh from volumetric data at a specified level.
    """
    verts, faces, _, _ = marching_cubes(volume=volume, level=level, spacing=spacing)
    return verts, faces


def plot_isosurface(Qx, xr, yr, zr, iso_value,
                          h=None,                  # (Ny, Nx) bathy depth (positive down)
                          plot_bathy=True,
                          bathy_stride=4,
                          Nz_out=None, zmin=None, zmax=None,
                          xmin=None, xmax=None, ymin=None, ymax=None,
                          z_positive_up=True,
                          base_color=[189, 20, 8]):
    """
    Compute & plot an isosurface of Qx using vertical interpolation, marching cubes, and 3D visualization.
    This function performs the following steps:
    1. Interpolates data from terrain-following coordinates (zr) to regular z-levels
    2. Applies marching cubes algorithm to extract isosurface geometry
    3. Renders the isosurface with Lambertian shading in matplotlib
    4. Optionally overlays bathymetry surface
    
    Parameters
    ----------
    Qx : ndarray, shape (Nz, Ny, Nx)
        3D scalar field to extract isosurface from
    xr : ndarray, shape (Nx,)
        X-coordinate array
    yr : ndarray, shape (Ny,)
        Y-coordinate array
    zr : ndarray, shape (Nz, Ny, Nx)
        Physical z-coordinates corresponding to Qx (terrain-following grid)
    iso_value : float
        Isosurface level value to extract
    h : ndarray, shape (Ny, Nx), optional
        Bathymetry depth (positive down). If provided, bathymetry surface is plotted.
    plot_bathy : bool, default True
        Whether to plot bathymetry surface overlay
    bathy_stride : int, default 4
        Stride for bathymetry mesh sampling (reduces point density for visualization)
    Nz_out : int, optional
        Number of output z-levels for interpolation. Defaults to 2*Nz
    zmin, zmax : float, optional
        Z-axis limits. Defaults to min/max of zr
    xmin, xmax : float, optional
        X-axis limits. Defaults to min/max of xr
    ymin, ymax : float, optional
        Y-axis limits. Defaults to min/max of yr
    z_positive_up : bool, default True
        If True, z-axis points upward; if False, downward
    base_color : array-like, shape (3,), default [189, 20, 8]
        RGB color for isosurface (values 0-255)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes3DSubplot
        3D axes object
    verts : ndarray, shape (N, 3)
        Isosurface vertex coordinates in index space
    faces : ndarray, shape (M, 3)
        Triangle face indices
    XYZ : tuple of ndarray
        Physical coordinates (X, Y, Z) of vertices
    z_levels : ndarray
        Regular z-level grid used for interpolation
    Qreg : ndarray, shape (Nz_out, Ny, Nx)
        Interpolated scalar field on regular z-levels
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
    ax.view_init(elev=30,azim=130) #30,130
    
    # Set aspect ratio
    ax.set_box_aspect((90/110, 1.0, 0.2))
    fig.tight_layout()
    plt.show()

    return fig, ax, (verts, faces), (X, Y, Z), z_levels, Qreg


def plot_two_isosurfaces(Qa, Qb, xr, yr, zr,
                         iso_a, iso_b,
                         h=None,
                         plot_bathy=True,
                         bathy_stride=4,
                         Nz_out=None, zmin=None, zmax=None,
                         xmin=None, xmax=None, ymin=None, ymax=None,
                         z_positive_up=True,
                         # appearance
                         color_a=(189/255, 20/255, 8/255),     # red
                         color_b=(30/255, 90/255, 170/255),    # blue
                         alpha_a=0.90,
                         alpha_b=0.65,
                         ls_azdeg=90, ls_altdeg=20):
    """
    Plot two isosurfaces from 3D volumetric data with optional bathymetry overlay.

    This function computes and visualizes two isosurfaces (at specified iso-values) 
    from 3D scalar fields defined on a sigma-coordinate grid. Both surfaces are 
    rendered with Gouraud shading and can be displayed alongside bathymetry data.

    Parameters
    ----------
    Qa : ndarray, shape (Nz, Ny, Nx)
        First scalar field on sigma-coordinate grid.
    Qb : ndarray, shape (Nz, Ny, Nx)
        Second scalar field on sigma-coordinate grid (must match Qa shape).
    xr : ndarray, shape (Nx,)
        X-coordinate array (1D, regularly or irregularly spaced).
    yr : ndarray, shape (Ny,)
        Y-coordinate array (1D, regularly or irregularly spaced).
    zr : ndarray, shape (Nz, Ny, Nx)
        Physical z-coordinates at each grid point.
    iso_a : float
        Iso-value for isosurface A.
    iso_b : float
        Iso-value for isosurface B.
    h : ndarray, shape (Ny, Nx), optional
        Bathymetry depth (positive down). If None, no bathymetry is plotted.
    plot_bathy : bool, default True
        Whether to plot bathymetry if h is provided.
    bathy_stride : int, default 4
        Stride for bathymetry grid sampling to reduce clutter.
    Nz_out : int, optional
        Number of regular z-levels for interpolation. If None, defaults to 2*Nz.
    zmin, zmax : float, optional
        Z-axis limits. If None, inferred from zr.
    xmin, xmax : float, optional
        X-axis limits. If None, inferred from xr.
    ymin, ymax : float, optional
        Y-axis limits. If None, inferred from yr.
    z_positive_up : bool, default True
        If True, positive z points upward; else positive z points downward.
    color_a : tuple, default (189/255, 20/255, 8/255)
        RGB color for isosurface A (red).
    color_b : tuple, default (30/255, 90/255, 170/255)
        RGB color for isosurface B (blue).
    alpha_a : float, default 0.90
        Transparency of isosurface A (0-1).
    alpha_b : float, default 0.65
        Transparency of isosurface B (0-1).
    ls_azdeg : float, default 90
        Light source azimuth angle (degrees).
    ls_altdeg : float, default 20
        Light source altitude angle (degrees).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes3DSubplot
        The 3D axes object.
    verts_a : ndarray
        Vertex coordinates of isosurface A from marching cubes.
    faces_a : ndarray
        Face indices of isosurface A from marching cubes.
    verts_b : ndarray
        Vertex coordinates of isosurface B from marching cubes.
    faces_b : ndarray
        Face indices of isosurface B from marching cubes.
    z_levels : ndarray
        Regular z-level grid used for interpolation.
    fields_reg : tuple of ndarray
        Interpolated fields (Qareg, Qbreg) on regular z-levels.

    Raises
    ------
    ValueError
        If Qb, zr, or h shapes don't match expected dimensions, or if all 
        interpolated values are NaN.

    Notes
    -----
    - Data is first interpolated onto regular z-levels using 1D linear interpolation 
        per (x,y) column, then isosurfaces are extracted via marching cubes algorithm.
    - Lighting is applied per-face using vertex normals from marching cubes with 
        intensity clipping for visual clarity.
    - 3D view is fixed at elev=40°, azim=130°; aspect ratio set for oceanographic data.
    """

    Nz, Ny, Nx = Qa.shape
    if Qb.shape != (Nz, Ny, Nx):
        raise ValueError(f"Qb must have same shape as Qa {(Nz,Ny,Nx)}; got {Qb.shape}")
    if zr.shape != (Nz, Ny, Nx):
        raise ValueError(f"zr must have shape {(Nz,Ny,Nx)}; got {zr.shape}")

    dx = float(xr[1] - xr[0])
    dy = float(yr[1] - yr[0])

    # --- limits / grids ---
    if zmin is None: zmin = np.nanmin(zr)
    if zmax is None: zmax = np.nanmax(zr)
    if xmin is None: xmin = np.nanmin(xr)
    if xmax is None: xmax = np.nanmax(xr)
    if ymin is None: ymin = np.nanmin(yr)
    if ymax is None: ymax = np.nanmax(yr)
    if Nz_out is None: Nz_out = Nz * 2

    z_levels = np.linspace(zmin, zmax, Nz_out)
    dz = float(z_levels[1] - z_levels[0])

    # --- interpolate BOTH fields onto regular z-levels ---
    Qareg = np.full((Nz_out, Ny, Nx), np.nan, dtype=float)
    Qbreg = np.full((Nz_out, Ny, Nx), np.nan, dtype=float)

    for j in range(Ny):
        for i in range(Nx):
            zcol = zr[:, j, i]
            a    = Qa[:, j, i]
            b    = Qb[:, j, i]

            m = np.isfinite(zcol) & np.isfinite(a) & np.isfinite(b)
            if m.sum() < 2:
                continue

            zc = zcol[m]
            ac = a[m]
            bc = b[m]

            idx = np.argsort(zc)
            zc = zc[idx]
            ac = ac[idx]
            bc = bc[idx]

            Qareg[:, j, i] = np.interp(z_levels, zc, ac, left=np.nan, right=np.nan)
            Qbreg[:, j, i] = np.interp(z_levels, zc, bc, left=np.nan, right=np.nan)

    def _work_volume(Qreg):
        Qwork = Qreg.copy()
        nan_mask = ~np.isfinite(Qwork)
        if np.all(nan_mask):
            raise ValueError("All values are NaN after interpolation.")
        finite_min = np.nanmin(Qwork)
        Qwork[nan_mask] = finite_min - 1e6
        return Qwork

    Qawork = _work_volume(Qareg)
    Qbwork = _work_volume(Qbreg)

    # --- marching cubes for each field ---
    verts_a, faces_a, normals_a, _ = marching_cubes(Qawork, level=iso_a, spacing=(dz, dy, dx))
    verts_b, faces_b, normals_b, _ = marching_cubes(Qbwork, level=iso_b, spacing=(dz, dy, dx))

    # Convert verts to physical coordinates
    x0 = float(xr[0]); y0 = float(yr[0]); z0 = float(z_levels[0])

    def _to_triangles(verts, faces):
        Z = verts[:, 0] + z0
        Y = verts[:, 1] + y0
        X = verts[:, 2] + x0
        XYZ = np.c_[X, Y, Z]
        tri = np.stack([XYZ[faces[:, 0]], XYZ[faces[:, 1]], XYZ[faces[:, 2]]], axis=1)
        return tri, normals_from_vertices(normals= None)

    # per-face normals from marching_cubes vertex normals
    def _face_normals(normals, faces):
        fn = normals[faces].mean(axis=1)
        nrm = np.linalg.norm(fn, axis=1)
        nrm[nrm == 0] = 1.0
        return fn / nrm[:, None]

    tri_a = np.stack([np.c_[verts_a[:,2]+x0, verts_a[:,1]+y0, verts_a[:,0]+z0][faces_a[:,0]],
                      np.c_[verts_a[:,2]+x0, verts_a[:,1]+y0, verts_a[:,0]+z0][faces_a[:,1]],
                      np.c_[verts_a[:,2]+x0, verts_a[:,1]+y0, verts_a[:,0]+z0][faces_a[:,2]]], axis=1)

    tri_b = np.stack([np.c_[verts_b[:,2]+x0, verts_b[:,1]+y0, verts_b[:,0]+z0][faces_b[:,0]],
                      np.c_[verts_b[:,2]+x0, verts_b[:,1]+y0, verts_b[:,0]+z0][faces_b[:,1]],
                      np.c_[verts_b[:,2]+x0, verts_b[:,1]+y0, verts_b[:,0]+z0][faces_b[:,2]]], axis=1)

    fn_a = _face_normals(normals_a, faces_a)
    fn_b = _face_normals(normals_b, faces_b)

    # --- lighting / shading ---
    ls = LightSource(azdeg=ls_azdeg, altdeg=ls_altdeg)
    light_dir = ls.direction

    def _shaded_facecolors(fn_unit, base_color):
        intensity = np.dot(fn_unit, light_dir)
        intensity = np.clip(intensity, 0.3, 1.0)
        base = np.array(base_color, dtype=float)
        return intensity[:, None] * base[None, :]

    facecolors_a = _shaded_facecolors(fn_a, color_a)
    facecolors_b = _shaded_facecolors(fn_b, color_b)

    # --- plot ---
    fig = plt.figure(figsize=(9, 7), dpi=300)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    # Bathymetry first (background)
    if plot_bathy and (h is not None):
        if h.shape != (Ny, Nx):
            raise ValueError(f"h must have shape (Ny,Nx)={(Ny,Nx)}, got {h.shape}")

        X2, Y2 = np.meshgrid(xr, yr)
        zb = (-h) if z_positive_up else (h)

        bathy = ax.plot_surface(
            X2[::bathy_stride, ::bathy_stride],
            Y2[::bathy_stride, ::bathy_stride],
            zb[::bathy_stride, ::bathy_stride],
            rstride=30, cstride=30, linewidth=0,
            alpha=0.8, shade=True, antialiased=True,
            color='orange'
        )
        bathy.set_zorder(0)

    # Isosurface A
    mesh_a = Poly3DCollection(tri_a, linewidths=0.0, alpha=alpha_a)
    mesh_a.set_facecolor(facecolors_a)
    mesh_a.set_edgecolor('none')
    mesh_a.set_linewidth(0.0)
    mesh_a.set_antialiased(False)
    mesh_a.set_zorder(19)
    ax.add_collection3d(mesh_a)

    # Isosurface B (slightly behind A in draw order)
    mesh_b = Poly3DCollection(tri_b, linewidths=0.0, alpha=alpha_b)
    mesh_b.set_facecolor(facecolors_b)
    mesh_b.set_edgecolor('none')
    mesh_b.set_linewidth(0.0)
    mesh_b.set_antialiased(False)
    mesh_b.set_zorder(20)
    ax.add_collection3d(mesh_b)

    # Labels (horizontal) + Z label as 2D text (robust)
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.set_xlabel("x [m]", fontsize=11, rotation=0)
    ax.set_ylabel("y [m]", fontsize=11, rotation=0)
    ax.xaxis.set_label_coords(0.5, -1.33)
    ax.yaxis.set_label_coords(1.05, 0.5)

    ax.set_zlabel("")
    ax.text2D(0.98, 0.67, "Depth [m]", transform=ax.transAxes,
              rotation=0, va="center", ha="left", fontsize=11)

    ax.tick_params(axis='z', pad=5)
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))

    # Limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    # View / aspect (match your previous)
    ax.view_init(elev=40, azim=130)
    ax.set_box_aspect((90/110, 1.0, 0.2))

    # IMPORTANT: don't use tight_layout for 3D
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.98)

    plt.show()

    return fig, ax, (verts_a, faces_a), (verts_b, faces_b), z_levels, (Qareg, Qbreg)
