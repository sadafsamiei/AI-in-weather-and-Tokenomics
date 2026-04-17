"""
Todo: 
    - save numpy array that generated map
    - outline the part that is not zero or set the zero part to black
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.patches as mpatches
import cartopy.feature as cfeature

from cartopy.crs import PlateCarree
from matplotlib import colors

from assets import Folders

VALUE_ANNOTATION_THRESHOLD = 0.1

def post_process_attribution(
        array_3d: np.ndarray, 
        grad_accumulation_strategy: str
    ):
    """
    Input:
        array_2d: 3D raw attribution_map 
        grad_accumulation_strategy: whether to keep directions of gradients or keep only magnitude.
    Performed steps:
        - Normalize values in range corresponding to "grad_accumulation_strategy": either [0,1] or [-1,1].
        - Generate norm object (colors.Normalize) for plot.
        - Define appropriate color map for plot.
    """
    eps = 1e-8  # for numerical stability
    min_, max_ = np.min(array_3d), np.max(array_3d)
    if grad_accumulation_strategy == "absolute":
        array_3d = (array_3d - min_) / (max_ - min_ + eps)
        norm = colors.Normalize(vmin=0, vmax=1)
        cmap = "Spectral_r"  # other possibilities: "seismic", "magma"
    elif grad_accumulation_strategy == "directional":
        array_3d = 2 * (array_3d - min_) / (max_ - min_ + eps) - 1
        norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        cmap = "RdBu"
    else:
        raise Exception("strategy not admissible")
    return array_3d, cmap, norm

def _visualize_attribution(
    array_2d: np.ndarray,
    cmap: str,
    norm: colors.Normalize,
    target_lat_lon: float,
):
    ny, nx = array_2d.shape

    # --- build edges, then centers from edges ---
    lon_e = np.linspace(0.0, 360.0, nx + 1, endpoint=True)
    lat_e = np.linspace(90.0, -90.0, ny + 1, endpoint=True)
    lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])   # shape (nx,)
    lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])   # shape (ny,)

    # target pixel center (indices provided)
    j = int(target_lat_lon[1])
    i = int(target_lat_lon[0])
    lon_deg = float(lon_c[j])
    lat_deg = float(lat_c[i])

    # projection centered on target to avoid seam issues
    proj = PlateCarree(central_longitude=lon_deg)
    fig, ax = plt.subplots(subplot_kw={"projection": proj})

    # plot
    pcm = ax.pcolormesh(
        lon_c, lat_c, array_2d,
        transform=proj, 
        cmap=cmap,
        norm=norm,
        shading="nearest"
    )
    LINES_COLOR = "black"
    ax.coastlines(color=LINES_COLOR, linewidth=0.05)
    ax.add_feature(cfeature.BORDERS, color=LINES_COLOR, linewidth=0.05)

    # --- crop window around the target center ---
    ratio = 0.5
    lon_off = 10.0
    lat_off = lon_off * ratio
    ax.set_extent(
        [lon_deg - lon_off, lon_deg + lon_off, lat_deg - lat_off, lat_deg + lat_off],
        crs=proj
    )

    NEG_THRESH = -0.5
    POS_THRESH = 0.1

    # --- annotate values at cell centers ---
    condition = (array_2d > POS_THRESH) | (array_2d < NEG_THRESH)
    ii, jj = np.where(condition)
    for yy, xx in zip(ii, jj):
        ax.text(
            lon_c[xx], lat_c[yy],
            f"{array_2d[yy, xx]:.2f}",
            color="black", ha="center", va="center",
            transform=proj, fontsize=1.5, zorder=5, clip_on=True
        )

    # --- outline the target pixel using edges computed from centers ---
    dlon = lon_e[1] - lon_e[0]        
    dlat = lat_e[0] - lat_e[1]        
    lon0 = lon_deg - 0.5 * dlon
    lon1 = lon_deg + 0.5 * dlon
    lat_bot = lat_deg - 0.5 * dlat
    lat_top = lat_deg + 0.5 * dlat

    COLOR = 'black'  # try white again
    rect = mpatches.Rectangle(
        (lon0, lat_bot), lon1 - lon0, lat_top - lat_bot,
        transform=proj, fill=False, edgecolor=COLOR, linewidth=0.7
    )
    ax.add_patch(rect)

    # still here in case it's needed in the future
    # ax.set_title(f"{var} — {smoother_name} — {explainer_name}", fontsize=5)
    # plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.01, shrink=0.5)
    # inset colorbar inside the plot
    
    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # cax = inset_axes(
    #     ax,
    #     width="10%",    # smaller width relative to the axes
    #     height="2%",    # very slim bar
    #     loc="lower left",
    #     borderpad=0.5   # tighter inside the axes
    # )
    # fig.colorbar(pcm, cax=cax, orientation="horizontal")
    # plt.close(fig)
    return fig
    
def _save_plot(fig, fig_name):
    fig.savefig(
        Folders.attributions_png + "/" + fig_name,
        dpi=1000,  
        bbox_inches="tight",
        pad_inches=0
    )  

def generate_attribution_viz(
        array_3d,
        cmap, norm,
        target_lat_lon,
        grad_accumulation_strategy,
        experiment_names
    ):
    """
    Takes care in one pass of:
    - Postprocessing the 3D attribution map.
    - Generate a visualization for each channel.
    - Logging the parameters and name of figure to file under /logs/results_index.json .
    """
    for c in range(array_3d.shape[0]):
        fig_name = experiment_names[c] + ".png"
        array_2d = array_3d[c]  # dims: (lat, lon), values: var

        fig = _visualize_attribution(
            array_2d,  
            cmap, norm,
            target_lat_lon
        )
        _save_plot(fig, fig_name)


    