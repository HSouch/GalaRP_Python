import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation





def animated_hexbin_plot(orbits, x_ind=1, y_ind=2, n_frames=100, **kwargs):
    """
    Create an animated hexbin plot of the given orbits.

    Parameters:
    - orbits: The orbits data.
    - x_ind: The index of the x-coordinate in the orbit data. Default is 1.
    - y_ind: The index of the y-coordinate in the orbit data. Default is 2.
    - n_frames: The number of frames in the animation. Default is 100.
    - **kwargs: Additional keyword arguments for customizing the plot.

    Returns:
    - None
    """
    
    outname = kwargs.get("outname", "animated_hexbin.gif")
    close = kwargs.get("close", True)
    cmap = kwargs.get("cmap", "viridis")

    figsize = kwargs.get("figsize", (10, 10))

    gridsize = kwargs.get("gridsize", 30)
    xextent = kwargs.get("xextent", (-40., 40.))
    xextent = (-xextent, xextent) if isinstance(xextent, (int, float)) else xextent
    yextent = kwargs.get("yextent", (-40., 40.))
    yextent = (-yextent, yextent) if isinstance(yextent, (int, float)) else yextent
    vextent = kwargs.get("vextent", (-100, 300))
    vextent = (-vextent, vextent) if isinstance(vextent, (int, float)) else vextent

    vmin, vmax = kwargs.get("vmin", 1), kwargs.get("vmax", 500)

    def add_labels(ax):
        ax[1][0].set_xlabel("X [kpc]")
        ax[0][0].set_ylabel("Y [kpc]")
        ax[1][0].set_ylabel("Z [kpc]")
        ax[0][1].set_xlabel("Z [kpc]")
        ax[0][1].set_ylabel(r"$V_z$ [km/ s$^{-1}$]")
        ax[1][1].set_xlabel("Time [Myr]")
        ax[1][1].set_ylabel(r"RP Profile [g/ (cm s$^2$)]")

    
    fig, ax = plt.subplots(2, 2, facecolor="white", figsize=figsize)

    imin = kwargs.get("imin", 0)
    imax = kwargs.get("imax", len(orbits.data.t) - 1)
    frames = np.linspace(imin, imax, n_frames).astype(int)

    x,y,z,vx,vy,vz = orbits.get_orbit_data(transposed=False)   

    
    ax[0][0].hexbin(x[0], y[0], bins="log", cmap=cmap, gridsize=(gridsize, gridsize),
                    extent=[xextent[0], xextent[1], xextent[0], xextent[1]], 
                    vmin=vmin, vmax=vmax, zorder = 5)
    ax[1][0].hexbin(x[0], z[0], bins="log", cmap=cmap, gridsize=(gridsize, gridsize),
                    extent=[xextent[0], xextent[1], yextent[0], yextent[1]], 
                    vmin=vmin, vmax=vmax, zorder = 5)
    ax[0][1].hexbin(z[0], vz[0], bins="log", cmap=cmap, gridsize=(gridsize, gridsize),
                    extent=[vextent[0], vextent[1], -400, 400], 
                    vmin=vmin, vmax=vmax, zorder = 5)
    
    
    add_labels(ax)

    plt.tight_layout()

    
    def animate(i):
        this_x, this_y, this_z = x[i], y[i], z[i]

        for axis in ax.flatten()[:]:
            axis.cla()

        ax[0][0].hexbin(x[i], y[i], bins="log", cmap=cmap, gridsize=(gridsize, gridsize),
                        extent=[xextent[0], xextent[1], xextent[0], xextent[1]], 
                        vmin=vmin, vmax=vmax, zorder = 5)
        
        
        ax[1][0].hexbin(x[i], z[i], bins="log", cmap=cmap, gridsize=(gridsize, gridsize),
                        extent=[xextent[0], xextent[1], yextent[0], yextent[1]], 
                        vmin=vmin, vmax=vmax, zorder = 5)
        ax[0][1].hexbin(z[i], vz[i], bins="log", cmap=cmap, gridsize=(gridsize, gridsize),
                        extent=[vextent[0], vextent[1], -400, 400], 
                        vmin=vmin, vmax=vmax, zorder = 5)
        ylims = ax[0][0].get_ylim()


        current_time = orbits.data.t[i].value
        ylims = ax[1][1].get_ylim()
        ax[1][1].plot([current_time, current_time], [ylims[0], ylims[1]], color="red", lw=1, ls="dashed")

        add_labels(ax)

    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=100)
    ani.save(outname, writer='pillow', fps=24)

    if close:
        plt.close(fig)

