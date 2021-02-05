"""
This module implements plotters for Fermi surfaces and Fermi slices.

TODO:
- Projections onto arbitrary surface
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib import cm
from monty.dev import requires
from monty.json import MSONable
from pymatgen import Spin
from pymatgen.symmetry.bandstructure import HighSymmKpath
from trimesh import transform_points

from ifermi.brillouin_zone import ReciprocalCell, ReciprocalSlice
from ifermi.fermi_surface import FermiSlice, FermiSurface

try:
    import mayavi.mlab as mlab
except ImportError:
    mlab = False

try:
    import kaleido
except ImportError:
    kaleido = False

try:
    from crystal_toolkit.core.scene import Lines, Scene, Spheres, Surface

    crystal_toolkit = True
except ImportError:
    crystal_toolkit = False

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
    aspectmode="data",
)

_plotly_label_style = dict(
    xshift=15, yshift=15, showarrow=False, font={"size": 20, "color": "black"}
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
_default_azimuth = 45.0
_default_elevation = 35.0


class FermiSurfacePlotter(MSONable):
    """
    Class to plot a FermiSurface.
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
        symprec: float = 1e-3,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get the high symmetry k-points and labels for the Fermi surface.

        Args:
            fermi_surface: A fermi surface.
            symprec: The symmetry precision in Angstrom.

        Returns:
            The high symmetry k-points and labels.
        """
        hskp = HighSymmKpath(fermi_surface.structure, symprec=symprec)
        labels, kpoints = list(zip(*hskp.kpath["kpoints"].items()))

        if isinstance(fermi_surface.reciprocal_space, ReciprocalCell):
            kpoints = kpoints_to_first_bz(np.array(kpoints))

        kpoints = np.dot(kpoints, fermi_surface.reciprocal_space.reciprocal_lattice)

        return kpoints, labels

    def get_plot(
        self,
        plot_type: str = "plotly",
        spin: Optional[Spin] = None,
        colors: Optional[Union[str, dict, list]] = None,
        azimuth: float = _default_azimuth,
        elevation: float = _default_elevation,
        **plot_kwargs,
    ):
        """
        Plot the Fermi surface.

        Args:
            plot_type: Method used for plotting. Valid options are: "matplotlib",
                "plotly", "mayavi", "crystal_toolkit".
            spin: Which spin channel to plot. By default plot both spin channels if
                available.
            colors: See the docstring for ``get_isosurfaces_and_colors()`` for the
                available options.
            azimuth: The azimuth of the viewpoint in degrees. i.e. the angle subtended
                by the position vector on a sphere projected on to the x-y plane.
            elevation: The zenith angle of the viewpoint in degrees, i.e. the angle
                subtended by the position vector and the z-axis.
            **plot_kwargs: Other keyword arguments supported by the individual plotting
                methods.
        """
        plot_kwargs.update(
            {"colors": colors, "spin": spin, "azimuth": azimuth, "elevation": elevation}
        )
        if plot_type == "matplotlib":
            plot = self.get_matplotlib_plot(**plot_kwargs)
        elif plot_type == "plotly":
            plot = self.get_plotly_plot(**plot_kwargs)
        elif plot_type == "mayavi":
            plot = self.get_mayavi_plot(**plot_kwargs)
        elif plot_type == "crystal_toolkit":
            plot = self.get_crystal_toolkit_plot(**plot_kwargs)
        else:
            types = ["matplotlib", "plotly", "mayavi", "crystal_toolkit"]
            error_msg = "Plot type not recognised, valid options: {}".format(types)
            raise ValueError(error_msg)
        return plot

    def get_isosurfaces_and_colors(
        self,
        plot_type: str = "plotly",
        spin: Optional[Spin] = None,
        colors: Optional[Union[str, dict, list]] = None,
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Any]:
        """
        Get the isosurfaces and colors to plot.

        Args:
            plot_type: Method used for plotting. Valid options are: "matplotlib",
                "plotly", "mayavi", "crystal_toolkit".
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

        if colors is None and plot_type == "plotly":
            colors = _get_plotly_colors(self.fermi_surface.isosurfaces, spin)
        else:
            colors = _get_random_colors(colors, self.fermi_surface.isosurfaces, spin)

        return isosurfaces, colors

    def get_matplotlib_plot(
        self,
        bz_linewidth: float = 0.9,
        spin: Optional[Spin] = None,
        colors: Optional[Union[str, dict, list]] = None,
        azimuth: float = _default_azimuth,
        elevation: float = _default_elevation,
    ):
        """
        Plot the Fermi surface using matplotlib.

        Args:
            bz_linewidth: Brillouin zone line width.
            spin: Which spin channel to plot. By default plot both spin channels if
                available.
            colors: See the docstring for ``get_isosurfaces_and_colors()`` for the
                available options.
            azimuth: The azimuth of the viewpoint in degrees. i.e. the angle subtended
                by the position vector on a sphere projected on to the x-y plane.
            elevation: The zenith angle of the viewpoint in degrees, i.e. the angle
                subtended by the position vector and the z-axis.

        Returns:
            matplotlib pyplot object.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d", proj_type="persp")

        isosurfaces, colors = self.get_isosurfaces_and_colors(
            spin=spin, colors=colors, plot_type="matplotlib"
        )

        # create a mesh for each electron band which has an isosurfaces at the Fermi
        # energy mesh data is generated by a marching cubes algorithm when the
        # FermiSurface object is created.
        for c, (verts, faces) in zip(colors, isosurfaces):
            x, y, z = verts.T
            ax.plot_trisurf(x, y, faces, z, facecolor=c, lw=1)

        # add the cell outline to the plot
        lines = Line3DCollection(
            self.reciprocal_space.lines, colors="k", linewidths=bz_linewidth
        )
        ax.add_collection3d(lines)

        for coords, label in zip(*self._symmetry_pts):
            ax.scatter(*coords, s=20, c="k", zorder=20)
            ax.text(*coords, "${}$".format(label), size=18, zorder=20)

        xlim, ylim, zlim = np.linalg.norm(self.rlat, axis=1) / 2
        ax.set(xlim=(-xlim, xlim), ylim=(-ylim, ylim), zlim=(-zlim, zlim))
        ax.view_init(elev=elevation, azim=azimuth)
        ax.axis("off")
        plt.tight_layout()

        return plt

    def get_plotly_plot(
        self,
        spin: Optional[Spin] = None,
        colors: Optional[Union[str, dict, list]] = None,
        azimuth: float = _default_azimuth,
        elevation: float = _default_elevation,
    ):
        """
        Plot the Fermi surface using plotly.

        Args:
            spin: Which spin channel to plot. By default plot both spin channels if
                available.
            colors: See the docstring for ``get_isosurfaces_and_colors()`` for the
                available options.
            azimuth: The azimuth of the viewpoint in degrees. i.e. the angle subtended
                by the position vector on a sphere projected on to the x-y plane.
            elevation: The zenith angle of the viewpoint in degrees, i.e. the angle
                subtended by the position vector and the z-axis.

        Returns:
            Plotly figure object.
        """
        import plotly.graph_objs as go
        from plotly.offline import init_notebook_mode

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
            x, y, z = verts.T
            i, j, k = faces.T
            trace = go.Mesh3d(x=x, y=y, z=z, color=c, opacity=1, i=i, j=j, k=k)
            meshes.append(trace)

        # add the cell outline to the plot
        for line in self.reciprocal_space.lines:
            x, y, z = line.T
            line_style = dict(color="black", width=3)
            trace = go.Scatter3d(x=x, y=y, z=z, mode="lines", line=line_style)
            meshes.append(trace)

        # plot high symmetry labels
        labels = ["${}$".format(i) for i in self._symmetry_pts[1]]
        x, y, z = self._symmetry_pts[0].T
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
            scene=scene, showlegend=False, margin=go.layout.Margin(l=0, r=0, b=0, t=0)
        )
        fig = go.Figure(data=meshes, layout=layout)
        fig.update_layout(scene_camera=_get_plotly_camera(azimuth, elevation))

        return fig

    @requires(mlab, "mayavi option requires mayavi to be installed.")
    def get_mayavi_plot(
        self,
        spin: Optional[Spin] = None,
        colors: Optional[Union[str, dict, list]] = None,
        azimuth: float = _default_azimuth,
        elevation: float = _default_elevation,
    ):
        """
        Plot the Fermi surface using mayavi.

        Args:
            spin: Which spin channel to plot. By default plot both spin channels if
                available.
            colors: See the docstring for ``get_isosurfaces_and_colors()`` for the
                available options.
            azimuth: The azimuth of the viewpoint in degrees. i.e. the angle subtended
                by the position vector on a sphere projected on to the x-y plane.
            elevation: The zenith angle of the viewpoint in degrees, i.e. the angle
                subtended by the position vector and the z-axis.

        Returns:
            mlab figure object.
        """
        from mlabtex import mlabtex

        mlab.figure(figure=None, bgcolor=(1, 1, 1), size=(800, 800))
        isosurfaces, colors = self.get_isosurfaces_and_colors(
            spin=spin, colors=colors, plot_type="mayavi"
        )

        for line in self.reciprocal_space.lines:
            x, y, z = line.T
            mlab.plot3d(x, y, z, **_mayavi_rs_style)

        for c, (verts, faces) in zip(colors, isosurfaces):
            x, y, z = verts.T
            mlab.triangular_mesh(x, y, z, faces, color=tuple(c), opacity=0.7)

        # latexify labels
        labels = ["${}$".format(i) for i in self._symmetry_pts[1]]
        for coords, label in zip(self._symmetry_pts[0], labels):
            mlabtex(*coords, label, **_mayavi_high_sym_label_style)

        mlab.view(azimuth=azimuth - 180, elevation=elevation - 90, distance="auto")

        return mlab

    @requires(
        crystal_toolkit,
        "crystal_toolkit option requires crystal_toolkit to be installed.",
    )
    def get_crystal_toolkit_plot(
        self,
        spin: Optional[Spin] = None,
        colors: Optional[Union[str, dict, list]] = None,
        opacity: float = 1.0,
        azimuth: float = _default_azimuth,
        elevation: float = _default_elevation,
    ) -> "Scene":
        """
        Get a crystal toolkit Scene showing the Fermi surface. The Scene can be
        displayed in an interactive web app using Crystal Toolkit, can be shown
        interactively in Jupyter Lab using the crystal-toolkit lab extension, or
        can be converted to JSON to store for future use.

        Args:
            spin: Which spin channel to plot. By default plot both spin channels if
                available.
            colors: See the docstring for ``get_isosurfaces_and_colors()`` for the
                available options.
            opacity: Opacity of surface. Note that due to limitations of WebGL,
                overlapping semi-transparent surfaces might result in visual artefacts.
            azimuth: The azimuth of the viewpoint in degrees. i.e. the angle subtended
                by the position vector on a sphere projected on to the x-y plane.
            elevation: The zenith angle of the viewpoint in degrees, i.e. the angle
                subtended by the position vector and the z-axis.

        Returns:
            Crystal-toolkit scene.
        """

        # The implementation here is very similar to the plotly implementation, except
        # the crystal toolkit scene is constructed using the scene primitives from
        # crystal toolkit (Spheres, Surface, Lines, etc.)

        scene_contents = []

        isosurfaces, colors = self.get_isosurfaces_and_colors(spin=spin, colors=colors)

        if isinstance(colors, np.ndarray):
            colors = (colors * 255).astype(int)
            colors = ["rgb({},{},{})".format(*c) for c in colors]

        # create a mesh for each electron band which has an isosurfaces at the Fermi
        # energy mesh data is generated by a marching cubes algorithm when the
        # FermiSurface object is created.
        surfaces = []
        for c, (verts, faces) in zip(colors, isosurfaces):
            positions = verts[faces].reshape(-1, 3).tolist()
            surface = Surface(positions=positions, color=c, opacity=opacity)
            surfaces.append(surface)
        fermi_surface = Scene("fermi_surface", contents=surfaces)
        scene_contents.append(fermi_surface)

        # add the cell outline to the plot
        lines = Lines(positions=list(self.reciprocal_space.lines.flatten()))
        # alternatively,
        # cylinders have finite width and are lighted, but no strong reason to choose
        # one over the other
        # cylinders = Cylinders(positionPairs=self.reciprocal_space.lines.tolist(),
        #                       radius=0.01, color="rgb(0,0,0)")
        scene_contents.append(lines)

        spheres = []
        for position, label in zip(self._symmetry_pts[0], self._symmetry_pts[1]):
            sphere = Spheres(
                positions=[list(position)],
                tooltip=label,
                radius=0.05,
                color="rgb(0, 0, 0)",
            )
            spheres.append(sphere)
        label_scene = Scene("labels", contents=spheres)
        scene_contents.append(label_scene)

        return Scene("ifermi", contents=scene_contents)


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

        kpoints = np.dot(
            kpoints, fermi_slice.reciprocal_slice.reciprocal_space.reciprocal_lattice
        )
        kpoints = transform_points(kpoints, fermi_slice.reciprocal_slice.transformation)

        # filter points that do not lie very close to the plane
        on_plane = np.where(np.abs(kpoints[:, 2]) < 1e-4)[0]
        kpoints = kpoints[on_plane]
        labels = [labels[i] for i in on_plane]

        return kpoints[:, :2], labels

    def get_plot(
        self,
        spin: Optional[Spin] = None,
        colors: Optional[Union[str, dict, list]] = "viridis",
    ):
        """
        Plot the Fermi slice.

        Args:
            spin: Which spin channel to plot. By default plot both spin channels if
                available.
            colors: The color specification. Valid options are:

                - A list of colors.
                - A dictionary of ``{Spin.up: color1, Spin.down: color2}``.
                - A string specifying which matplotlib colormap to use. See
                  https://matplotlib.org/tutorials/colors/colormaps.html for more
                  information.
                - ``None``, in which case the colors will be chosen randomly.

        Returns:
            matplotlib pyplot object.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

        # get a rotation matrix that will align the longest slice length along the
        # x-axis
        rotation = _get_rotation(self.fermi_slice.reciprocal_slice)

        slices, colors = self.get_slices_and_colors(spin=spin, colors=colors)

        # create a mesh for each electron band which has an isosurfaces at the Fermi
        # energy mesh data is generated by a marching cubes algorithm when the
        # FermiSurface object is created.
        for c, a_slice in zip(colors, slices):
            lines = LineCollection(np.dot(a_slice, rotation), colors=c, linewidth=2)
            ax.add_collection(lines)

        # add the cell outline to the plot
        rotated_lines = np.dot(self.reciprocal_slice.lines, rotation)
        lines = LineCollection(rotated_lines, colors="k", linewidth=1)
        ax.add_collection(lines)

        for coords, label in zip(*self._symmetry_pts):
            coords = np.dot(coords, rotation)
            ax.scatter(*coords, s=20, c="k")
            label = label.replace(r"\Gamma", r"$\Gamma$")
            ax.text(*coords, " " + label, size=18, zorder=10)

        ax.autoscale(enable=True)
        ax.axis("equal")
        ax.axis("off")

        return plt

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

        colors = _get_random_colors(colors, self.fermi_slice.slices, spin)

        return slices, colors


def show_plot(plot: Any):
    """Display a plot.

    Args:
        plot: A plot object from ``FermiSurfacePlotter.get_plot()``. Supports matplotlib
            pyplot objects, plotly figure objects, and mlab figure objects.
    """
    plot_type = get_plot_type(plot)

    if plot_type == "matplotlib":
        plot.show()
    elif plot_type == "plotly":
        from plotly.offline import plot as show_plotly

        show_plotly(plot, include_mathjax="cdn", filename="fermi-surface.html")
    elif plot_type == "mayavi":
        plot.show()


def save_plot(
    plot: Any,
    filename: Union[Path, str],
    scale: float = 4,
):
    """Save a plot to file.

    Args:
        plot: A plot object from ``FermiSurfacePlotter.get_plot()``. Supports matplotlib
            pyplot objects, plotly figure objects, and mlab figure objects.
        filename: The output filename.
        scale: Scale for the figure size. Increases resolution but does not change the
            relative size of the figure and text.
    """
    plot_type = get_plot_type(plot)
    filename = str(filename)
    if plot_type == "matplotlib":
        # default dpi is ~100
        plot.savefig(filename, dpi=scale * 100, bbox_inches="tight")
    elif plot_type == "plotly":
        if kaleido is None:
            raise ValueError(
                "kaleido package required to save static ploty images\n"
                "please install it using:\npip install kaleido"
            )
        plot.write_image(
            filename,
            engine="kaleido",
            scale=scale,
        )
    elif plot_type == "mayavi":
        plot.savefig(filename, magnification=scale)


def _get_plotly_camera(azimuth: float, elevation: float) -> Dict[str, Dict[str, float]]:
    """Get plotly viewpoint from azimuth and elevation."""
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)
    norm = np.linalg.norm([1.25, 1.25, 1.25])  # default plotly vector distance
    x = np.sin(azimuth) * np.cos(elevation) * norm
    y = np.cos(azimuth) * np.cos(elevation) * norm
    z = np.sin(elevation) * norm
    return dict(
        up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=x, y=y, z=z)
    )


def get_plot_type(plot: Any) -> str:
    """Gets the plot type.

    Args:
        plot: A plot object from ``FermiSurfacePlotter.get_plot()``. Supports matplotlib
            pyplot objects, plotly figure objects, and mlab figure objects.

    Returns:
        The plot type. Current options are "matplotlib", "mayavi", and "plotly".
    """
    from plotly.graph_objs import Figure

    if isinstance(plot, Figure):
        return "plotly"
    elif hasattr(plot, "__name__"):
        if "matplotlib" in plot.__name__:
            return "matplotlib"
        elif "mayavi" in plot.__name__:
            return "mayavi"
    raise ValueError("Unrecognised plot type.")


def kpoints_to_first_bz(kpoints: np.ndarray, tol: float = 1e-5) -> np.ndarray:
    """Translate fractional k-points to the first Brillouin zone.

    I.e. all k-points will have fractional coordinates:
        -0.5 <= fractional coordinates < 0.5

    Args:
        kpoints: The k-points in fractional coordinates.
        tol: Numerical tolerance.

    Returns:
        The translated k-points.
    """
    kp = kpoints - np.round(kpoints)

    # account for small rounding errors for 0.5
    round_dp = int(np.log10(1 / tol))
    krounded = np.round(kp, round_dp)

    kp[krounded == -0.5] = 0.5
    return kp


def _get_random_colors(
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


def _get_plotly_colors(
    objects: Dict[Spin, List[Any]],
    spins: List[Spin],
) -> Any:
    import plotly.express as px

    n_objects = sum([len(objects[spin]) for spin in spins])

    if n_objects < len(px.colors.qualitative.Prism):
        colors = px.colors.qualitative.Prism
    else:
        colors = []
        i = n_objects // len(px.colors.qualitative.Prism) + 1
        for count in range(i):
            colors.append(px.colors.qualitative.Prism)

    return colors


def _get_rotation(reciprocal_slice: ReciprocalSlice) -> np.ndarray:
    """
    Get a rotation matrix that aligns the longest slice length along the x axis.

    Args:
        reciprocal_slice: A reciprocal slice.

    Returns:
        The transformation matrix as 2x2 array.
    """
    line_vectors = reciprocal_slice.lines[:, 0, :] - reciprocal_slice.lines[:, 1, :]
    line_lengths = np.linalg.norm(line_vectors, axis=-1)
    longest_line = line_vectors[np.argmax(line_lengths)]

    longest_line_norm = longest_line / np.linalg.norm(longest_line)
    dotp = np.dot(longest_line_norm, [1, 0])
    angle = np.arccos(dotp)

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    return rotation
