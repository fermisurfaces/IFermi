"""
This module implements plotters for Fermi surfaces and Fermi slices.
"""
from dataclasses import dataclass

from matplotlib.colors import Colormap
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
from ifermi.kpoints import kpoints_to_first_bz

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
_default_vector_spacing = 0.2
_default_colormap = "viridis"


@dataclass
class FermiSurfacePlotData:
    isosurfaces: List[Tuple[np.ndarray, np.ndarray, int]]
    azimuth: float
    elevation: float
    colors: Optional[List[Tuple[int, int, int]]]
    projections: List[np.ndarray]
    arrows: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    projection_colormap: Optional[Colormap]
    arrow_colormap: Optional[Colormap]
    cmin: Optional[float]
    cmax: Optional[float]


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
        fermi_surface: FermiSurface, symprec: float = 1e-3
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
        color_projection: Union[str, bool] = True,
        vector_projection: Union[str, bool] = False,
        projection_axis: Optional[Tuple[int, int, int]] = None,
        vector_spacing: float = _default_vector_spacing,
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
        vnorm: Optional[float] = None,
        hide_surface: bool = False,
        **plot_kwargs,
    ):
        """
        Plot the Fermi surface.

        Args:
            plot_type: Method used for plotting. Valid options are: "matplotlib",
                "plotly", "mayavi", "crystal_toolkit".
            spin: Which spin channel to plot. By default plot both spin channels if
                available.
            azimuth: The azimuth of the viewpoint in degrees. i.e. the angle subtended
                by the position vector on a sphere projected on to the x-y plane.
            elevation: The zenith angle of the viewpoint in degrees, i.e. the angle
                subtended by the position vector and the z-axis.
            colors: The color specification for the iso-surfaces. Valid options are:

                - A single color to use for all Fermi surfaces, specified as a tuple of
                  rgb values from 0 to 1. E.g., red would be ``(1, 0, 0)``.
                - A list of colors, specified as above.
                - A dictionary of ``{Spin.up: color1, Spin.down: color2}``, where the
                  colors are specified as above.
                - A string specifying which matplotlib colormap to use. See
                  https://matplotlib.org/tutorials/colors/colormaps.html for more
                  information.
                - ``None``, in which case the default colors will be used.

            color_projection: Whether to use the projections to color the Fermi surface.
                If the projections is a vector then the norm of the projections will be
                used. Note, this will only take effect if the Fermi surface has
                projections. If set to True, the viridis colormap will be used.
                Alternative colormaps can be selected by setting ``vector_projection``
                to a matplotlib colormap name. This setting will override the ``colors``
                option. For vector projections, the arrows are colored according to the
                norm of the projections by default. If used in combination with the
                ``projection_axis`` option, the color will be determined by the dot
                product of the projections with the projections axis.
            vector_projection: Whether to plot arrows for vector projections. Note, this
                will only take effect if the Fermi surface has vector projections. If
                set to True, the viridis colormap will be used. Alternative colormaps
                can be selected by setting ``vector_projection`` to a matplotlib
                colormap name. By default, the arrows are colored according to the norm
                of the projections. If used in combination with the ``projection_axis``
                option, the color will be determined by the dot product of the
                projections with the projections axis.
            projection_axis: Projection axis that can be used to calculate the color of
                vector projects. If None, the norm of the projections will be used,
                otherwise the color will be determined by the dot product of the
                projections with the projections axis. Only has an effect when used with
                the ``vector_projection`` option.
            vector_spacing: The rough spacing between arrows. Uses a custom algorithm
                for resampling the Fermi surface to ensure that arrows are not too close
                together. Only has an effect when used with the ``vector_projection``
                option.
            cmin: Minimum intensity for normalising projection colors (including
                projection vector colors).Only has an effect when used with
                ``color_projection`` or ``vector_projection`` options.
            cmax: Maximum intensity for normalising projection colors (including
                projection vector colors). Only has an effect when used with
                ``color_projection`` or ``vector_projection`` options.
            vnorm: The value by which to normalize the vector lengths. For example,
                spin projections should typically have a norm of 1 whereas group
                velocity projections can have larger or smaller norms depending on the
                structure. By changing this number, the size of the vectors will be
                scaled. Note that the projections of two materials can only be compared
                quantitatively if a fixed values is used for both plots. Only has an
                effect when used with the ``vector_projection`` option.
            hide_surface: Whether to hide the Fermi surface. Only recommended in
                combination with the ``vector_projection`` option.
            **plot_kwargs: Other keyword arguments supported by the individual plotting
                methods.
        """
        plot_data = self._get_plot_data(
            spin=spin,
            azimuth=azimuth,
            elevation=elevation,
            colors=colors,
            color_projection=color_projection,
            vector_projection=vector_projection,
            projection_axis=projection_axis,
            vector_spacing=vector_spacing,
            cmin=cmin,
            cmax=cmax,
            vnorm=vnorm,
            hide_surface=hide_surface,
        )
        if plot_type == "matplotlib":
            plot = self.get_matplotlib_plot(plot_data, **plot_kwargs)
        elif plot_type == "plotly":
            plot = self.get_plotly_plot(plot_data, **plot_kwargs)
        elif plot_type == "mayavi":
            plot = self.get_mayavi_plot(plot_data, **plot_kwargs)
        elif plot_type == "crystal_toolkit":
            plot = self.get_crystal_toolkit_plot(plot_data, **plot_kwargs)
        else:
            types = ["matplotlib", "plotly", "mayavi", "crystal_toolkit"]
            error_msg = "Plot type not recognised, valid options: {}".format(types)
            raise ValueError(error_msg)
        return plot

    def _get_plot_data(
        self,
        spin: Optional[Spin] = None,
        azimuth: float = _default_azimuth,
        elevation: float = _default_elevation,
        colors: Optional[Union[str, dict, list]] = None,
        color_projection: Union[str, bool] = True,
        vector_projection: Union[str, bool] = False,
        projection_axis: Optional[Tuple[int, int, int]] = None,
        vector_spacing: float = _default_vector_spacing,
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
        vnorm: Optional[float] = None,
        hide_surface: bool = False,
    ) -> FermiSurfacePlotData:
        """
        Get the the Fermi surface plot data.

        See ``FermiSurfacePlotter.get_plot()`` for more details.

        Returns:
            The Fermi surface plot data.
        """
        if not spin:
            spin = list(self.fermi_surface.isosurfaces.keys())
        elif isinstance(spin, Spin):
            spin = [spin]

        isosurfaces = []
        if not hide_surface:
            for s in spin:
                isosurfaces.extend(self.fermi_surface.isosurfaces[s])

        projections = []
        projection_colormap = None
        if self.fermi_surface.projections is not None:
            # always calculate projections if they are present so we can determine
            # cmin and cmax. These are also be used for arrows and it is critical that
            # cmin and cmax are the same for projections and arrow color scales (even
            # if the colormap used is different)
            projections, projection_colormap = _get_face_projections(
                self.fermi_surface, spin, projection_axis=projection_axis
            )
            if isinstance(color_projection, str):
                projection_colormap = cm.get_cmap(color_projection)
            else:
                projection_colormap = cm.get_cmap(_default_colormap)
            cmin, cmax = get_projection_limits(projections)

        if not color_projection:
            colors = get_isosurface_colors(colors, self.fermi_surface.isosurfaces, spin)
            projections = []
            cmin = None
            cmax = None
        else:
            colors = None

        arrows = []
        arrow_colormap = None
        if vector_projection is not None and self.fermi_surface.projections is not None:
            arrows = _get_arrows(
                self.fermi_surface,
                spin,
                projection_axis=projection_axis,
                vector_spacing=vector_spacing,
                vnorm=vnorm,
            )
            if isinstance(vector_projection, str):
                arrow_colormap = cm.get_cmap(vector_projection)
            else:
                arrow_colormap = cm.get_cmap(_default_colormap)

        return FermiSurfacePlotData(
            isosurfaces=isosurfaces,
            azimuth=azimuth,
            elevation=elevation,
            colors=colors,
            projections=projections,
            arrows=arrows,
            projection_colormap=projection_colormap,
            arrow_colormap=arrow_colormap,
            cmin=cmin,
            cmax=cmax,
        )

    def get_matplotlib_plot(
        self,
        plot_data: FermiSurfacePlotData,
        bz_linewidth: float = 0.9,
    ):
        """
        Plot the Fermi surface using matplotlib.

        Args:
            plot_data: The plot data.
            bz_linewidth: Brillouin zone line width.

        Returns:
            matplotlib pyplot object.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d", proj_type="persp")

        isosurfaces, colors = self._get_plot_data(
            spin=spin, colors=colors, plot_type="matplotlib"
        )

        # create a mesh for each electron band which has an isosurfaces at the Fermi
        # energy mesh data is generated by a marching cubes algorithm when the
        # FermiSurface object is created.
        for c, (verts, faces, band_idx) in zip(colors, isosurfaces):
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

    def get_plotly_plot(self, plot_data: FermiSurfacePlotData):
        """
        Plot the Fermi surface using plotly.

        Args:
            plot_data: The data to plot.

        Returns:
            Plotly figure object.
        """
        import plotly.graph_objs as go
        from plotly.offline import init_notebook_mode

        init_notebook_mode(connected=True)

        if isinstance(colors, np.ndarray):
            colors = (colors * 255).astype(int)
            colors = ["rgb({},{},{})".format(*c) for c in colors]

        # create a mesh for each electron band which has an isosurfaces at the Fermi
        # energy mesh data is generated by a marching cubes algorithm when the
        # FermiSurface object is created.
        meshes = []
        if plot_data.projections:
            cmin, cmax = get_projection_limits(plot_data.projections)
            for (verts, faces, _), face_projections in zip(plot_data.isosurfaces, plot_data.projections):
                x, y, z = verts.T
                i, j, k = faces.T
                trace = go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    opacity=1,
                    i=i,
                    j=j,
                    k=k,
                    intensity=face_projections,
                    colorscale=colors,
                    intensitymode='cell',
                    cmin=cmin,
                    cmax=cmax
                )
                meshes.append(trace)
        else:
            for c, (verts, faces, band_idx) in zip(colors, plot_data.isosurfaces):
                x, y, z = verts.T
                i, j, k = faces.T
                trace = go.Mesh3d(x=x, y=y, z=z, color=c, opacity=1, i=i, j=j, k=k)
                meshes.append(trace)

        for starts, ends, arrow_colors in plot_data.arrows:
            for start, end, arrow_color in zip(starts, ends, arrow_colors):
                meshes.extend(plotly_arrow(start, end, arrow_color))

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
        camera = _get_plotly_camera(plot_data.azimuth, plot_data.elevation)
        fig.update_layout(scene_camera=camera)

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
            colors: See the docstring for ``_get_plot_data()`` for the available options.
            azimuth: The azimuth of the viewpoint in degrees. i.e. the angle subtended
                by the position vector on a sphere projected on to the x-y plane.
            elevation: The zenith angle of the viewpoint in degrees, i.e. the angle
                subtended by the position vector and the z-axis.

        Returns:
            mlab figure object.
        """
        from mlabtex import mlabtex

        mlab.figure(figure=None, bgcolor=(1, 1, 1), size=(800, 800))
        isosurfaces, colors = self._get_plot_data(
            spin=spin, colors=colors, plot_type="mayavi"
        )

        for line in self.reciprocal_space.lines:
            x, y, z = line.T
            mlab.plot3d(x, y, z, **_mayavi_rs_style)

        for c, (verts, faces, band_idx) in zip(colors, isosurfaces):
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
            colors: See the docstring for ``_get_plot_data()`` for the available options.
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

        isosurfaces, colors = self._get_plot_data(
            spin=spin, colors=colors
        )

        if isinstance(colors, np.ndarray):
            colors = (colors * 255).astype(int)
            colors = ["rgb({},{},{})".format(*c) for c in colors]

        # create a mesh for each electron band which has an isosurfaces at the Fermi
        # energy mesh data is generated by a marching cubes algorithm when the
        # FermiSurface object is created.
        surfaces = []
        for c, (verts, faces, band_idx) in zip(colors, isosurfaces):
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
        colors: Optional[Union[str, dict, list]] = _default_colormap,
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

        for c, (a_slice, band_idx) in zip(colors, slices):
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
    ) -> Tuple[List[Tuple[np.ndarray, int]], Any]:
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

        colors = get_isosurface_colors(colors, self.fermi_slice.slices, spin)

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


def save_plot(plot: Any, filename: Union[Path, str], scale: float = 4):
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
        plot.write_image(filename, engine="kaleido", scale=scale)
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


def get_isosurface_colors(
    colors: Optional[Union[str, dict, list]],
    objects: Dict[Spin, List[Any]],
    spins: List[Spin],
) -> List[Tuple[float, float, float]]:
    """
    Get colors for each Fermi surface.

    Args:
        colors: The color specification. Valid options are:

            - A single color to use for all Fermi surfaces, specified as a tuple of rgb
              values from 0 to 1. E.g., red would be ``(1, 0, 0)``.
            - A list of colors, specified as above.
            - A dictionary of ``{Spin.up: color1, Spin.down: color2}``, where the colors
              are specified as above.
            - A string specifying which matplotlib colormap to use. See
              https://matplotlib.org/tutorials/colors/colormaps.html for more
              information.
            - ``None``, in which case the default colors will be used.
        objects: The iso-surfaces or 2d slices for which colors will be generated.
            Should be specified as a dict of ``{spin: spin_objects}`` where
            spin_objects is a list of objects.
        spins: A list of spins for which colors will be generated.

    Returns:
        The colors as a list of tuples, where each color is specified as the rgb values
        from 0 to 1. E.g., red would be ``(1, 0, 0)``.
    """
    n_objects = sum([len(objects[spin]) for spin in spins])

    if isinstance(colors, (tuple, list, np.ndarray)):
        if isinstance(colors[0], (tuple, list, np.ndarray)):
            # colors is a list of colors
            cc = list(colors) * (len(colors) // n_objects + 1)
            return cc[:n_objects]
        else:
            # colors is a single color specification
            return [colors] * n_objects

    elif isinstance(colors, dict):
        if len(colors) < len(spins):
            raise ValueError(
                "colors dict must have same number of spin channels as spins to plot"
            )
        return [colors[s] for s in spins for _ in objects[s]]

    elif isinstance(colors, str):
        # get rid of alpha channel
        return [i[:3] for i in cm.get_cmap(colors)(np.linspace(0, 1, n_objects))]

    else:
        from plotly.colors import qualitative, unlabel_rgb, unconvert_from_RGB_255
        cc = qualitative.Prism * (len(qualitative.Prism) // n_objects + 1)
        return [unconvert_from_RGB_255(unlabel_rgb(c)) for c in cc[:n_objects]]


def plotly_arrow(start: np.ndarray, stop: np.ndarray, color: np.ndarray) -> Tuple[Any, Any]:
    import plotly.graph_objs as go
    cone_length = 0.08

    vector = stop - start
    vector /= np.linalg.norm(vector)

    color = color_to_plotly(color)
    colorscale = [[0, color], [1, color]]

    line = go.Scatter3d(
        x=[start[0], stop[0]],
        y=[start[1], stop[1]],
        z=[start[2], stop[2]],
        mode="lines",
        line={"width": 6, "color": color},
        showlegend=False
    )
    cone = go.Cone(
        x=[stop[0]],
        y=[stop[1]],
        z=[stop[2]],
        u=[vector[0]],
        v=[vector[1]],
        w=[vector[2]],
        showscale=False,
        colorscale=colorscale,
        sizemode="absolute",
        sizeref=cone_length,
        anchor="cm"
    )
    return line, cone


def resample_mesh(vertices: np.ndarray, faces: np.ndarray, grid_size: float):
    face_verts = vertices[faces]
    centers = face_verts.mean(axis=1)
    min_coords = np.min(centers, axis=0)
    max_coords = np.max(centers, axis=0)

    lengths = (max_coords - min_coords)
    min_coords -= lengths * 0.2
    max_coords += lengths * 0.2

    n_grid = np.ceil((max_coords - min_coords) / grid_size).astype(int)

    center_idxs = np.arange(len(centers))

    selected_faces = []
    for cell_image in np.ndindex(tuple(n_grid)):
        cell_image = np.array(cell_image)
        cell_min = min_coords + cell_image * grid_size
        cell_max = min_coords + (cell_image + 1) * grid_size

        # find centers that fall within the
        within = np.all(centers > cell_min, axis=1) & np.all(centers < cell_max, axis=1)

        if not np.any(within):
            continue

        # get the indexes of those centers
        within_idx = center_idxs[within]

        # of these, find the center that is closest to the center of the cell
        distances = np.linalg.norm((cell_max + cell_min) / 2 - centers[within], axis=1)
        select_idx = within_idx[np.argmin(distances)]

        selected_faces.append(select_idx)

    return np.array(selected_faces)


def _get_face_projections(
    fermi_surface: FermiSurface,
    spins: List[Spin],
    projection_axis: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    Get projections and projections colormap.

    Args:
        fermi_surface: The fermi surface containing the isosurfaces and projections.
        spins: A list of spins to extract the projections for.
        projection_axis: Projection axis that can be used to calculate the color of
            vector projects. If None, the norm of the projections will be used,
            otherwise the color will be determined by the dot product of the projections
            with the projections axis.

    Returns:
        The projections as a list of numpy arrays (one array for each isosurface), where
        the shape of the array is (nfaces, ).
    """
    face_projections = []
    for spin in spins:
        for iso_projections in fermi_surface.projections[spin]:
            if iso_projections.ndim == 2:
                if projection_axis is None:
                    # color projections by the norm of the projections
                    iso_projections = np.linalg.norm(iso_projections, axis=1)
                else:
                    # color projections by projecting the vector onto axis
                    iso_projections = np.dot(iso_projections, projection_axis)

            face_projections.append(iso_projections)

    return face_projections


def _get_arrows(
    fermi_surface: FermiSurface,
    spins: List[Spin],
    vector_spacing: float = _default_vector_spacing,
    vnorm: Optional[float] = None,
    projection_axis: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Get arrows from vector projections.

    Args:
        fermi_surface: The fermi surface containing the isosurfaces and projections.
        spins: Spin channels from which to extract arrows.
        vector_spacing: The rough spacing between arrows. Uses a custom algorithm for
            resampling the Fermi surface to ensure that arrows are not too close
            together.
        vnorm: The value by which to normalize the vector lengths. For example,
            spin projections should typically have a norm of 1 whereas group velocity
            projections can have larger or smaller norms depending on the structure.
            By changing this number, the size of the vectors will be scaled. Note that
            the projections of two materials can only be compared quantitatively if a
            fixed values is used for both plots.
        projection_axis: Projection axis that can be used to calculate the color of
            vector projects. If None, the norm of the projections will be used,
            otherwise the color will be determined by the dot product of the projections
            with the projections axis.

    Returns:
        The arrows, as a list of (starts, stops, intenties) for each face. The
        starts and stops are numpy arrays with the shape (narrows, 3) and intensities
        is a numpy array with the shape (narrows, ). The intensities are used
        to color the arrows during plotting.
    """
    norms = []
    centers = []
    intensity = []
    vectors = []
    for spin in spins:
        for (vertices, faces, _), iso_projections in zip(
            fermi_surface.isosurfaces[spin],
            fermi_surface.projections[spin]
        ):
            if iso_projections.ndim != 2:
                continue

            face_idx = resample_mesh(vertices, faces, vector_spacing)

            faces = faces[face_idx]
            iso_projections = iso_projections[face_idx]

            # get the center of each of face in cartesian coords
            face_verts = vertices[faces]
            centers.append(face_verts.mean(axis=1))

            vectors.append(iso_projections)
            norms.append(np.linalg.norm(iso_projections, axis=1))

            if projection_axis is None:
                # projections intensity is the norm of the projections
                intensity.append(norms[-1])
            else:
                # get projections intensity from projections of the vector onto axis
                intensity.append(np.dot(iso_projections, projection_axis))

    # get vector norm
    if vnorm is None:
        vnorm = np.max([np.max(x) for x in norms])

    arrows = []
    for face_vectors, face_centers, face_intensity in zip(vectors, centers, intensity):
        face_vectors *= 0.14 / vnorm  # 0.14 is magic scaling factor for vector length
        start = face_centers - face_vectors / 2
        stop = start + face_vectors
        # color = cmap((face_intensity - cmin) / (cmax - cmin))
        arrows.append((start, stop, face_intensity))

    return arrows


def get_projection_limits(
    projections: List[np.ndarray],
    cmin: Optional[float] = None,
    cmax: Optional[float] = None
) -> Tuple[float, float]:
    if cmax is None:
        cmax = np.max([np.max(x) for x in projections])

    if cmin is None:
        cmin = np.min([np.min(x) for x in projections])

    return cmin, cmax


def color_to_plotly(color):
    if isinstance(color, (tuple, list)):
        color = "rgb({},{},{})".format(*(np.array(color) * 255).astype(int))
    return color


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
