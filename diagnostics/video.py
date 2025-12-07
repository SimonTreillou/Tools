import matplotlib.animation as animation
from IPython.display import Video
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# write_map_to_video: Generates and saves an animated video from a sequence of 2D data arrays using matplotlib.
# ---------------------------------------------------------------

def write_map_to_video(var,time,path,x=None,y=None,title=r'$\omega$ ($s^{-1})$',fps=5,
                       vmin=-0.1,vmax=0.1,cmap="RdBu_r",label="Vorticity",norm=None,
                       xlab=r'$x$ (m)', ylab=r'$y$ (m)',vline=None):
    """
    Generates and saves an animated video from a sequence of 2D data arrays using matplotlib.

    Parameters
    ----------
    var : np.ndarray
        3D array of shape (n_frames, n_y, n_x) representing the variable to visualize at each time step.
    time : array-like
        1D array of time values corresponding to each frame in `var`.
    path : str
        Output filename (without extension) for the saved video. The video will be saved in the directory indicated by `path`.
    title : str, optional
        Title for the plot, can include LaTeX formatting. Default is r'$\omega$ ($s^{-1})$'.
    fps : int, optional
        Frames per second for the output video. Default is 5.
    vmin : float, optional
        Minimum value for color normalization. Default is -0.1.
    vmax : float, optional
        Maximum value for color normalization. Default is 0.1.
    cmap : str, optional
        Colormap to use for the plot. Default is "RdBu_r".
    label : str, optional
        Label for the colorbar. Default is "Vorticity".

    Notes
    -----
    - Requires matplotlib and ffmpeg to be installed.
    - The video is saved in MP4 format using the MPEG4 codec.
    - The function creates a color mesh plot for each frame and updates the colorbar and title accordingly.
    """
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title="Video", artist='Simon Treillou', comment='Allez le TFC') 
    writer = FFMpegWriter(fps=fps, metadata=metadata, codec='mpeg4')

    if norm is None:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    if x is None:
        x = np.arange(var.shape[2]) # !!! Not working with coordinates !!!
    if y is None:
        y = np.arange(var.shape[1])

    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(8,8), dpi=300)
    pcm = ax.pcolormesh(var[0], cmap=cmap, shading="auto", norm=norm)
    cbar = fig.colorbar(pcm, ax=ax, label=label)
    if vline is not None:
        ax.vlines(vline, ymin=y[0], ymax=y[-1], colors='k', linestyles='--')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    tit=ax.set_title(title + f' at t = {time[0]:.2f} s',loc='left')

    def update(frame):
        pcm.set_array(var[frame].ravel())
        tit.set_text(title + f' at t = {time[frame]:.2f} s')
        return pcm, tit

    ani = animation.FuncAnimation(fig, update, frames=var.shape[0], interval=100, blit=True)

    # Save animation to file
    ani.save(path+'.mp4', writer=writer)
    
def write_plot_to_video(var,time,path,x=None,title=r'$\omega$ ($s^{-1})$',
    fps=5,label="Vorticity",ymin=None,ymax=None,xmin=None,xmax=None,xlab=r'$x$ (m)',
    ylab=r'$y$ (m)',vline=None,xscale="linear",yscale="linear",static_curves=None,):
    """
    Generates and saves an animated video from a sequence of 1D data arrays using matplotlib.

    Parameters
    ----------
    var : np.ndarray
        2D array of shape (n_frames, n_x) representing the variable to visualize at each time step.
    time : array-like
        1D array of time values corresponding to each frame in `var`.
    path : str
        Output filename (without extension) for the saved video.
    title : str, optional
        Title for the plot. Default is r'$\omega$ ($s^{-1})$'.
    fps : int, optional
        Frames per second. Default is 5.
    label : str, optional
        Label for the animated curve. Default is "Vorticity".
    static_curves : list[dict], optional
        List of static curves to add once at initialization. Each dict supports:
        - 'x': array-like (required)
        - 'y': array-like (required)
        - 'label': str (optional)
        - 'style': dict of Line2D kwargs (optional), e.g. {'color':'k','ls':'--'}

    Notes
    -----
    - Requires matplotlib and ffmpeg.
    - Saves MP4 using MPEG4 codec.
    """
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title="Video", artist='Simon Treillou', comment='Allez le TFC')
    writer = FFMpegWriter(fps=fps, metadata=metadata, codec='mpeg4')

    if x is None:
        x = np.arange(var.shape[1])
    if ymin is None:
        ymin = np.nanmin(var)
    if ymax is None:
        ymax = np.nanmax(var)
    if xmin is None:
        xmin = np.nanmin(x)
    if xmax is None:
        xmax = np.nanmax(x)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    # Static curves (non-animated), plotted once
    if static_curves:
        for sc in static_curves:
            sx = np.asarray(sc.get('x'))
            sy = np.asarray(sc.get('y'))
            slabel = sc.get('label', None)
            sstyle = sc.get('style', {})
            ax.plot(sx, sy, label=slabel, **sstyle)

    # Main animated line
    line, = ax.plot(x, var[0], label=label)

    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    if vline is not None:
        ax.vlines(vline, ymin=ymin, ymax=ymax, colors='k', linestyles='--')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    # Show legend if any labels were provided
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()

    tit = ax.set_title(title + f' at t = {time[0]:.2f} s', loc='left')

    def update(frame):
        line.set_xdata(x)
        line.set_ydata(var[frame])
        tit.set_text(title + f' at t = {time[frame]:.2f} s')
        return line, tit

    ani = animation.FuncAnimation(fig, update, frames=var.shape[0], interval=100, blit=True)
    ani.save(path + '.mp4', writer=writer)