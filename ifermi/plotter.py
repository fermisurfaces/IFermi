"""
This module implements a plotter for the Fermi-Surface of a material
todo:
* Remap into Brillioun zone (Wigner Seitz cell)
* Get Latex working for labels
* Do projections onto arbitrary surface
* Comment more
* Think about classes/methods, maybe restructure depending on sumo layout
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib import cm
from monty.dev import requires
from monty.json import MSONable
from trimesh import transform_points

from ifermi.brillouin_zone import ReciprocalCell
from ifermi.fermi_surface import FermiSlice, FermiSurface
from pymatgen import Spin
from pymatgen.symmetry.bandstructure import HighSymmKpath

try:
    import plotly
except ImportError:
    plotly = False

try:
    import mayavi.mlab as mlab
except ImportError:
    mlab = False

_plotly_scene = dict(
    xaxis=dict(
        backgroundcolor="rgb(255, 255, 255)",
        title="",
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
        showticklabels=False,
    ),
    yaxis=dict(
        backgroundcolor="rgb(255, 255, 255)",
        title="",
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
        showticklabels=False,
    ),
    zaxis=dict(
        backgroundcolor="rgb(255, 255, 255)",
        title="",
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
        showticklabels=False,
    ),
)

_plotly_label_style = dict(
    xshift=15, yshift=15, showarrow=False, font={"size": 20}
)

_mayavi_high_sym_label_style = {
    "color": (0, 0, 0),
    "scale": 0.1,
    "orientation": (90.0, 0.0, 0.0),
}

_mayavi_rs_style = {
    "color": (0.0, 0.0, 0.0),
    "tube_radius": 0.005,
    "representation": "surface",
}


class FermiSurfacePlotter(MSONable):
    """
    Class to plot FermiSurface.
    """

    def __init__(self, fermi_surface: FermiSurface):
        """
        Args:
            fermi_surface: A FermiSurface object.
        """
        self.fermi_surface = fermi_surface
        self.reciprocal_space = fermi_surface.reciprocal_space
        self.rlat = self.reciprocal_space.reciprocal_lattice
        self._symmetry_pts = self.get_symmetry_points(fermi_surface)

    @staticmethod
    def get_symmetry_points(
        fermi_surface: FermiSurface,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get the high symmetry k-points and labels for the Fermi surface.

        Args:
            fermi_surface: A fermi surface.

        Returns:
            The high symmetry k-points and labels.
        """
        hskp = HighSymmKpath(fermi_surface.structure)
        labels, kpoints = list(zip(*hskp.kpath["kpoints"].items()))

        if isinstance(fermi_surface.reciprocal_space, ReciprocalCell):
            kpoints = kpoints_to_first_bz(np.array(kpoints))

        kpoints = hskp.rec_lattice.get_cartesian_coords(kpoints)

        return kpoints, labels

    def plot(
        self,
        plot_type: str = "plotly",
        interactive: bool = True,
        filename: str = "fermi_surface.png",
        spin: Optional[Spin] = None,
        colors: Optional[Union[str, dict, list]] = None,
        **plot_kwargs,
    ):
        """
        Plot the Fermi surface and save the image to a file.

        Args:
            plot_type: Method used for plotting. Valid options are: "matplotlib",
                "plotly", "mayavi".
            interactive: Whether to enable interactive plots.
            filename: Output filename.
            spin: Which spin channel to plot. By default plot both spin channels if
                available.
            colors: See the docstring for ``get_isosurfaces_and_colors()`` for the
                available options.
            **plot_kwargs: Other keyword arguments supported by the individual plotting
                methods.
        """
        plot_kwargs.update(
            dict(filename=filename, interactive=interactive, colors=colors, spin=spin)
        )
        if plot_type == "mpl":
            self.plot_matplotlib(**plot_kwargs)
        elif plot_type == "plotly":
            self.plot_plotly(**plot_kwargs)
        elif plot_type == "mayavi":
            self.plot_mayavi(**plot_kwargs)
        else:
            types = ["mpl", "plotly", "mayavi"]
            error_msg = "Plot type not recognised, valid options: {}".format(types)
            raise ValueError(error_msg)

    def get_isosurfaces_and_colors(
        self,
        spin: Optional[Spin] = None,
        colors: Optional[Union[str, dict, list]] = None,
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Any]:
        """
        Get the isosurfaces and colors to plot.

        Args:
            spin: Which spin channel to select. By default will return the isosurfaces
                for both spin channels if available.
            colors: The color specification. Valid options are:
                - A list of colors.
                - A dictionary of ``{Spin.up: color1, Spin.down: color2}``.
                - A string specifying which matplotlib colormap to use. See
                  https://matplotlib.org/tutorials/colors/colormaps.html for more
                  information.
                - ``None``, in which case the colors will be chosen randomly.

        Returns:
            The isosurfaces and colors as a tuple.
        """
        if not spin:
            spin = list(self.fermi_surface.isosurfaces.keys())
        elif isinstance(spin, Spin):
            spin = [spin]

        isosurfaces = []
        for s in spin:
            isosurfaces.extend(self.fermi_surface.isosurfaces[s])

        colors = _get_colors(colors, self.fermi_surface.isosurfaces, spin)

        return isosurfaces, colors

    def plot_matplotlib(
        self,
        interactive: bool = True,
        bz_linewidth: float = 0.9,
        spin: Optional[Spin] = None,
        colors: Optional[Union[str, dict, list]] = None,
        title: str = None,
        filename: str = "fermi_surface.png",
    ):
        """
        Plot the Fermi surface using matplotlib.

        Args:
            interactive: Whether to enable interactive plots.
            bz_linewidth: Brillouin zone line width.
            spin: Which spin channel to plot. By default plot both spin channels if
                available.
            colors: See the docstring for ``get_isosurfaces_and_colors()`` for the
                available options.
            title: The title of the plot.
            filename: The output file name.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        isosurfaces, colors = self.get_isosurfaces_and_colors(spin=spin, colors=colors)

        # create a mesh for each electron band which has an isosurfaces at the Fermi
        # energy mesh data is generated by a marching cubes algorithm when the
        # FermiSurface object is created.
        for c, (verts, faces) in zip(colors, isosurfaces):
            x, y, z = zip(*verts)
            ax.plot_trisurf(x, y, faces, z, facecolor=c, lw=1)

        # add the cell outline to the plot
        lines = Line3DCollection(
            self.reciprocal_space.lines, colors="k", linewidths=bz_linewidth
        )
        ax.add_collection3d(lines)

        for coords, label in zip(*self._symmetry_pts):
            ax.scatter(*coords, s=10, c="k")
            ax.text(*coords, label, size=15, zorder=1)

        if title is not None:
            plt.title(title)

        xlim, ylim, zlim = np.linalg.norm(self.rlat, axis=1) / 2
        ax.set(xlim=(-xlim, xlim), ylim=(-ylim, ylim), zlim=(-zlim, zlim))

        ax.axis("off")
        plt.tight_layout()

        if interactive:
            plt.show()
        else:
            plt.savefig(filename, dpi=300)

    @requires(plotly, "plotly option requires plotly to be installed.")
    def plot_plotly(
        self,
        interactive: bool = True,
        spin: Optional[Spin] = None,
        colors: Optional[Union[str, dict, list]] = None,
        filename: str = "fermi_surface.png",
    ):
        """
        Plot the Fermi surface using plotly.

        Args:
            interactive: Whether to enable interactive plots.
            spin: Which spin channel to plot. By default plot both spin channels if
                available.
            colors: See the docstring for ``get_isosurfaces_and_colors()`` for the
                available options.
            filename: The output file name.
        """
        from plotly.offline import init_notebook_mode, plot
        import plotly.graph_objs as go

        init_notebook_mode(connected=True)
        isosurfaces, colors = self.get_isosurfaces_and_colors(spin=spin, colors=colors)

        if isinstance(colors, np.ndarray):
            colors = (colors * 255).astype(int)
            colors = ["rgb({},{},{})".format(*c) for c in colors]

        # create a mesh for each electron band which has an isosurfaces at the Fermi
        # energy mesh data is generated by a marching cubes algorithm when the
        # FermiSurface object is created.
        meshes = []
        for c, (verts, faces) in zip(colors, isosurfaces):
            x, y, z = zip(*verts)
            i, j, k = ([triplet[c] for triplet in faces] for c in range(3))
            trace = go.Mesh3d(x=x, y=y, z=z, color=c, opacity=1, i=i, j=j, k=k)
            meshes.append(trace)

        # add the cell outline to the plot
        for line in self.reciprocal_space.lines:
            x, y, z = zip(*line)
            line_style = dict(color="black", width=3)
            trace = go.Scatter3d(x=x, y=y, z=z, mode="lines", line=line_style)
            meshes.append(trace)

        # plot high symmetry labels
        # labels = [i.replace(r"\Gamma", "\u0393") for i in self._symmetry_pts[1]]
        labels = ["${}$".format(i) for i in self._symmetry_pts[1]]
        x, y, z = zip(*self._symmetry_pts[0])
        marker_style = dict(size=5, color="black")
        trace = go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=marker_style)
        meshes.append(trace)

        annotations = []
        for label, (x, y, z) in zip(labels, self._symmetry_pts[0]):
            # annotations always appear on top of the plot
            annotations.append(dict(x=x, y=y, z=z, text=label, **_plotly_label_style))
        scene = _plotly_scene.copy()
        scene["annotations"] = annotations

        # Specify plot parameters
        layout = go.Layout(
            scene=scene,
            showlegend=False,
            title=go.layout.Title(text="", xref="paper", x=0),
        )
        fig = go.Figure(data=meshes, layout=layout)

        if interactive:
            plot(fig, include_mathjax="cdn")
        else:
            plotly.io.write_image(fig, str(filename), width=600, height=600, scale=5)

    @requires(mlab, "mayavi option requires mayavi to be installed.")
    def plot_mayavi(
        self,
        interactive: bool = True,
        spin: Optional[Spin] = None,
        colors: Optional[Union[str, dict, list]] = None,
        filename: str = "fermi_surface.png",
    ):
        """
        Plot the Fermi surface using mayavi.

        Args:
            interactive: Whether to enable interactive plots.
            spin: Which spin channel to plot. By default plot both spin channels if
                available.
            colors: See the docstring for ``get_isosurfaces_and_colors()`` for the
                available options.
            filename: The output file name.
        """
        from mlabtex import mlabtex

        mlab.figure(figure=None, bgcolor=(1, 1, 1), size=(800, 800))
        isosurfaces, colors = self.get_isosurfaces_and_colors(spin=spin, colors=colors)

        for line in self.reciprocal_space.lines:
            x, y, z = zip(*line)
            mlab.plot3d(x, y, z, **_mayavi_rs_style)

        for c, (verts, faces) in zip(colors, isosurfaces):
            x, y, z = zip(*verts)
            mlab.triangular_mesh(x, y, z, faces, color=tuple(c), opacity=0.7)

        # latexify labels
        labels = ["${}$".format(i) for i in self._symmetry_pts[1]]
        for coords, label in zip(self._symmetry_pts[0], labels):
            mlabtex(*coords, label, **_mayavi_high_sym_label_style)

        if isinstance(self.reciprocal_space, ReciprocalCell):
            mlab.view(azimuth=0, elevation=60, distance=12)
        else:
            mlab.view(azimuth=235, elevation=60, distance=12)

        if interactive:
            mlab.show()
        else:
            mlab.savefig(str(filename), figure=mlab.gcf())


class FermiSlicePlotter(object):
    """
    Class to plot 2D slices through a FermiSurface.
    """

    def __init__(self, fermi_slice: FermiSlice):
        """
        Initialize a FermiSurfacePlotter.

        Args:
            fermi_slice: A slice through a Fermi surface.
        """
        self.fermi_slice = fermi_slice
        self.reciprocal_slice = fermi_slice.reciprocal_slice
        self._symmetry_pts = self.get_symmetry_points(fermi_slice)

    @staticmethod
    def get_symmetry_points(fermi_slice: FermiSlice) -> Tuple[np.ndarray, List[str]]:
        """
        Get the high symmetry k-points and labels for the Fermi slice.

        Args:
            fermi_slice: A fermi slice.

        Returns:
            The high symmetry k-points and labels for points that lie on the slice.
        """
        hskp = HighSymmKpath(fermi_slice.structure)
        labels, kpoints = list(zip(*hskp.kpath["kpoints"].items()))

        if isinstance(fermi_slice.reciprocal_slice.reciprocal_space, ReciprocalCell):
            kpoints = kpoints_to_first_bz(np.array(kpoints))

        kpoints = hskp.rec_lattice.get_cartesian_coords(kpoints)
        kpoints = transform_points(kpoints, fermi_slice.reciprocal_slice.transformation)

        # filter points that do not lie very close to the plane
        on_plane = np.where(np.abs(kpoints[:, 2]) < 1e-4)[0]
        kpoints = kpoints[on_plane]
        labels = [labels[i] for i in on_plane]

        return kpoints[:, :2], labels

    def plot(
        self,
        filename: str = "fermi_slice.png",
        spin: Optional[Spin] = None,
        colors: Optional[Union[str, dict, list]] = "viridis",
        show: bool = False,
    ):
        """
        Plot the Fermi surface and save the image to a file.

        Args:
            filename: Output filename.
            spin: Which spin channel to plot. By default plot both spin channels if
                available.
            colors: The color specification. Valid options are:
                - A list of colors.
                - A dictionary of ``{Spin.up: color1, Spin.down: color2}``.
                - A string specifying which matplotlib colormap to use. See
                  https://matplotlib.org/tutorials/colors/colormaps.html for more
                  information.
                - ``None``, in which case the colors will be chosen randomly.
            show: Show the plot before saving to file.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

        slices, colors = self.get_slices_and_colors(spin=spin, colors=colors)

        # create a mesh for each electron band which has an isosurfaces at the Fermi
        # energy mesh data is generated by a marching cubes algorithm when the
        # FermiSurface object is created.
        for c, a_slice in zip(colors, slices):
            lines = LineCollection(a_slice, colors=c, linewidth=2)
            ax.add_collection(lines)

        # add the cell outline to the plot
        lines = LineCollection(self.reciprocal_slice.lines, colors="k", linewidth=1)
        ax.add_collection(lines)

        for coords, label in zip(*self._symmetry_pts):
            ax.scatter(*coords, s=10, c="k")
            label = label.replace(r"\Gamma", r"$\Gamma$")
            ax.text(*coords, " " + label, size=15, zorder=1)

        ax.autoscale(enable=True)
        ax.axis("equal")
        ax.axis("off")

        if show:
            plt.show()
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    def get_slices_and_colors(
        self,
        spin: Optional[Spin] = None,
        colors: Optional[Union[str, dict, list]] = None,
    ) -> Tuple[List[np.ndarray], Any]:
        """
        Get the isosurfaces and colors to plot.

        Args:
            spin: Which spin channel to select. By default will return the slices
                for both spin channels if available.
            colors: The color specification. Valid options are:
                - A list of colors.
                - A dictionary of ``{Spin.up: color1, Spin.down: color2}``.
                - A string specifying which matplotlib colormap to use. See
                  https://matplotlib.org/tutorials/colors/colormaps.html for more
                  information.
                - ``None``, in which case the colors will be chosen randomly.

        Returns:
            The isosurfaces and colors as a tuple.
        """
        if not spin:
            spin = list(self.fermi_slice.slices.keys())
        elif isinstance(spin, Spin):
            spin = [spin]

        slices = []
        for s in spin:
            slices.extend(self.fermi_slice.slices[s])

        colors = _get_colors(colors, self.fermi_slice.slices, spin)

        return slices, colors


def kpoints_to_first_bz(kpoints: np.ndarray, tol=1e-5) -> np.ndarray:
    """Translate fractional k-points to the first Brillouin zone.

    I.e. all k-points will have fractional coordinates:
        -0.5 <= fractional coordinates < 0.5

    Args:
        kpoints: The k-points in fractional coordinates.

    Returns:
        The translated k-points.
    """
    kp = kpoints - np.round(kpoints)

    # account for small rounding errors for 0.5
    round_dp = int(np.log10(1 / tol))
    krounded = np.round(kp, round_dp)

    kp[krounded == -0.5] = 0.5
    return kp


def _get_colors(
    colors: Optional[Union[str, dict, list]],
    objects: Dict[Spin, List[Any]],
    spins: List[Spin],
) -> Any:
    """
    Plot the Fermi surface using matplotlib.

    Args:
        colors: See the docstring for ``get_isosurfaces_and_colors()`` for the
            available options.
    """
    n_objects = sum([len(objects[spin]) for spin in spins])

    if isinstance(colors, dict):
        if len(colors) < len(spins):
            raise ValueError(
                "colors dictionary should have the same number of spin channels as"
                "spins to plot"
            )
        colors = []
        for s in spins:
            colors.extend([colors[s]] * len(objects[s]))

    elif isinstance(colors, str):
        cmap = cm.get_cmap(colors)
        colors = cmap(np.linspace(0, 1, n_objects))

    elif colors is None:
        colors = np.random.random((n_objects, 3))

    return colors
