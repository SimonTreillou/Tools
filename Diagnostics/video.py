import matplotlib.animation as animation
from IPython.display import Video
import numpy as np
import matplotlib.pyplot as plt

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# write_map_to_video: Generates and saves an animated video from a sequence of 2D data arrays using matplotlib.
# ---------------------------------------------------------------

def write_map_to_video(var,time,path,title=r'$\omega$ ($s^{-1})$',fps=5,vmin=-0.1,vmax=0.1,cmap="RdBu_r",label="Vorticity"):
    """
    Generates and saves an animated video from a sequence of 2D data arrays using matplotlib.

    Parameters
    ----------
    var : np.ndarray
        3D array of shape (n_frames, n_y, n_x) representing the variable to visualize at each time step.
    time : array-like
        1D array of time values corresponding to each frame in `var`.
    path : str
        Output filename (without extension) for the saved video. The video will be saved in the './Videos/' directory.
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

    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(8,8), dpi=300)
    pcm = ax.pcolormesh(var[0], cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(pcm, ax=ax, label=label)
    cbar.set_clim(vmin, vmax)  # fix limits for the colorbar
    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r'$y$ (m)')
    tit=ax.set_title(title + f' at t = {time[0]:.2f} s',loc='left')

    def update(frame):
        pcm.set_array(var[frame].ravel())
        tit.set_text(title + f' at t = {time[frame]:.2f} s')
        return pcm, tit

    ani = animation.FuncAnimation(fig, update, frames=var.shape[0], interval=100, blit=True)

    # Save animation to file
    ani.save('./Videos/'+path+'.mp4', writer=writer)