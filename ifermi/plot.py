"""Tools to plot FermiSurface and FermiSlice objects."""

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib.colors import Colormap, Normalize
from monty.dev import requires
from pymatgen.electronic_structure.core import Spin

from ifermi.defaults import AZIMUTH, COLORMAP, ELEVATION, SCALE, SYMPREC, VECTOR_SPACING
from ifermi.slice import FermiSlice
from ifermi.surface import FermiSurface

try:
    import mayavi.mlab as mlab
except ImportError:
    mlab = False

try:
    import kaleido
except ImportError:
    kaleido = False

try:
    import crystal_toolkit
except ImportError:
    crystal_toolkit = False


# define plotly default styles
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
_plotly_bz_style = {"line": {"color": "black", "width": 3}}
_plotly_sym_pt_style = {"marker": {"size": 5, "color": "black"}}
_plotly_sym_label_style = dict(
    xshift=15, yshift=15, showarrow=False, font={"size": 20, "color": "black"}
)
_plotly_cbar_style = {"lenmode": "fraction", "len": 0.5, "tickfont": {"size": 15}}

# define mayavi default styles
_mayavi_sym_label_style = {
    "color": (0, 0, 0),
    "scale": 0.1,
    "orientation": (90.0, 0.0, 0.0),
}
_mayavi_rs_style = {
    "color": (0.0, 0.0, 0.0),
    "tube_radius": 0.005,
    "representation": "surface",
}

# define matplotlib default styles
_mpl_cbar_style = {"shrink": 0.5}
_mpl_bz_style = {"linewidth": 1, "color": "k"}
_mpl_arrow_style = {
    "angles": "xy",
    "scale_units": "xy",
    "scale": 1,
    "zorder": 10,
    "units": "dots",
    "width": 10,
    "pivot": "tail",
}

_mpl_sym_pt_style = {"s": 20, "c": "k", "zorder": 20}
_mpl_sym_label_style = {"size": 16, "zorder": 20}

__all__ = [
    "FermiSlicePlotter",
    "FermiSurfacePlotter",
    "save_plot",
    "show_plot",
    "get_plot_type",
    "get_isosurface_colors",
    "plotly_arrow",
    "rgb_to_plotly",
    "cmap_to_mayavi",
    "cmap_to_plotly",
    "get_segment_arrows",
    "get_face_arrows",
]


@dataclass
class _FermiSurfacePlotData:
    isosurfaces: List[Tuple[np.ndarray, np.ndarray]]
    azimuth: float
    elevation: float
    colors: List[Tuple[int, int, int]]
    properties: List[np.ndarray]
    arrows: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    properties_colormap: Optional[Colormap]
    arrow_colormap: Optional[Colormap]
    cmin: Optional[float]
    cmax: Optional[float]
    hide_labels: bool
    hide_cell: bool


@dataclass
class _FermiSlicePlotData:
    slices: List[np.ndarray]
    colors: List[Tuple[int, int, int]]
    properties: List[np.ndarray]
    arrows: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    properties_colormap: Optional[Colormap]
    arrow_colormap: Optional[Colormap]
    cmin: Optional[float]
    cmax: Optional[float]
    hide_labels: bool
    hide_cell: bool


class FermiSurfacePlotter:
    """Class to plot a FermiSurface.

    Args:
        fermi_surface: A FermiSurface object.
        symprec: The symmetry precision in Angstrom for determining the high
            symmetry k-point labels.
    """

    def __init__(self, fermi_surface: FermiSurface, symprec: float = SYMPREC):
        self.fermi_surface = fermi_surface
        self.reciprocal_space = fermi_surface.reciprocal_space
        self.rlat = self.reciprocal_space.reciprocal_lattice
        self._symmetry_pts = self.get_symmetry_points(fermi_surface, symprec=symprec)

    @staticmethod
    def get_symmetry_points(
        fermi_surface: FermiSurface, symprec: float = SYMPREC
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get the high symmetry k-points and labels for the Fermi surface.

        Args:
            fermi_surface: A fermi surface.
            symprec: The symmetry precision in Angstrom.

        Returns:
            The high symmetry k-points and labels.
        """
        from pymatgen.symmetry.bandstructure import HighSymmKpath

        from ifermi.brillouin_zone import WignerSeitzCell
        from ifermi.kpoints import kpoints_to_first_bz

        hskp = HighSymmKpath(fermi_surface.structure, symprec=symprec)
        labels, kpoints = list(zip(*hskp.kpath["kpoints"].items()))

        if not np.allclose(
            hskp.prim.lattice.matrix, fermi_surface.structure.lattice.matrix, 1e-5
        ):
            warnings.warn("Structure does not match expected primitive cell")

        if not isinstance(fermi_surface.reciprocal_space, WignerSeitzCell):
            kpoints = kpoints_to_first_bz(np.array(kpoints))

        kpoints = np.dot(kpoints, fermi_surface.reciprocal_space.reciprocal_lattice)

        return kpoints, labels

    def get_plot(
        self,
        plot_type: str = "plotly",
        spin: Optional[Spin] = None,
        colors: Optional[Union[str, dict, list]] = None,
        azimuth: float = AZIMUTH,
        elevation: float = ELEVATION,
        color_properties: Union[str, bool] = True,
        vector_properties: Union[str, bool] = False,
        projection_axis: Optional[Tuple[int, int, int]] = None,
        vector_spacing: float = VECTOR_SPACING,
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
        vnorm: Optional[float] = None,
        hide_surface: bool = False,
        hide_labels: bool = False,
        hide_cell: bool = False,
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

            color_properties: Whether to use the properties to color the Fermi surface.
                If the properties is a vector then the norm of the properties will be
                used. Note, this will only take effect if the Fermi surface has
                properties. If set to True, the viridis colormap will be used.
                Alternative colormaps can be selected by setting ``color_properties``
                to a matplotlib colormap name. This setting will override the ``colors``
                option. For vector properties, the arrows are colored according to the
                norm of the properties by default. If used in combination with the
                ``projection_axis`` option, the color will be determined by the dot
                product of the properties with the projection axis.
            vector_properties: Whether to plot arrows for vector properties. Note, this
                will only take effect if the Fermi surface has vector properties. If
                set to True, the viridis colormap will be used. Alternative colormaps
                can be selected by setting ``vector_properties`` to a matplotlib
                colormap name. By default, the arrows are colored according to the norm
                of the properties. If used in combination with the ``projection_axis``
                option, the color will be determined by the dot product of the
                properties with the projection axis.
            projection_axis: Projection axis that can be used to calculate the color of
                vector properties. If None, the norm of the properties will be used,
                otherwise the color will be determined by the dot product of the
                properties with the projection axis. Only has an effect when used with
                the ``vector_properties`` option.
            vector_spacing: The rough spacing between arrows. Uses a custom algorithm
                for resampling the Fermi surface to ensure that arrows are not too close
                together. Only has an effect when used with the ``vector_properties``
                option.
            cmin: Minimum intensity for normalising properties colors (including
                vector colors). Only has an effect when used with
                ``color_properties`` or ``vector_properties`` options.
            cmax: Maximum intensity for normalising properties colors (including
                vector colors). Only has an effect when used with
                ``color_properties`` or ``vector_properties`` options.
            vnorm: The value by which to normalize the vector lengths. For example,
                spin properties should typically have a norm of 1 whereas group
                velocity properties can have larger or smaller norms depending on the
                structure. By changing this number, the size of the vectors will be
                scaled. Note that the properties of two materials can only be compared
                quantitatively if a fixed values is used for both plots. Only has an
                effect when used with the ``vector_properties`` option.
            hide_surface: Whether to hide the Fermi surface. Only recommended in
                combination with the ``vector_properties`` option.
            hide_labels: Whether to show the high-symmetry k-point labels.
            hide_cell: Whether to show the reciprocal cell boundary.
            **plot_kwargs: Other keyword arguments supported by the individual plotting
                methods.
        """
        plot_data = self._get_plot_data(
            spin=spin,
            azimuth=azimuth,
            elevation=elevation,
            colors=colors,
            color_properties=color_properties,
            vector_properties=vector_properties,
            projection_axis=projection_axis,
            vector_spacing=vector_spacing,
            cmin=cmin,
            cmax=cmax,
            vnorm=vnorm,
            hide_surface=hide_surface,
            hide_labels=hide_labels,
            hide_cell=hide_cell,
        )
        if plot_type == "matplotlib":
            plot = self._get_matplotlib_plot(plot_data, **plot_kwargs)
        elif plot_type == "plotly":
            plot = self._get_plotly_plot(plot_data, **plot_kwargs)
        elif plot_type == "mayavi":
            plot = self._get_mayavi_plot(plot_data, **plot_kwargs)
        elif plot_type == "crystal_toolkit":
            plot = self._get_crystal_toolkit_plot(plot_data, **plot_kwargs)
        else:
            types = ["matplotlib", "plotly", "mayavi", "crystal_toolkit"]
            error_msg = "Plot type not recognised, valid options: {}".format(types)
            raise ValueError(error_msg)
        return plot

    def _get_plot_data(
        self,
        spin: Optional[Spin] = None,
        azimuth: float = AZIMUTH,
        elevation: float = ELEVATION,
        colors: Optional[Union[str, dict, list]] = None,
        color_properties: Union[str, bool] = True,
        vector_properties: Union[str, bool] = False,
        projection_axis: Optional[Tuple[int, int, int]] = None,
        vector_spacing: float = VECTOR_SPACING,
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
        vnorm: Optional[float] = None,
        hide_surface: bool = False,
        hide_labels: bool = False,
        hide_cell: bool = False,
    ) -> _FermiSurfacePlotData:
        """
        Get the the Fermi surface plot data.

        See ``FermiSurfacePlotter.get_plot()`` for more details.

        Returns:
            The Fermi surface plot data.
        """
        from matplotlib.cm import get_cmap

        if not spin:
            spin = self.fermi_surface.spins
        elif isinstance(spin, Spin):
            spin = [spin]

        isosurfaces = []
        if not hide_surface:
            isosurfaces = self.fermi_surface.all_vertices_faces(spins=spin)

        properties = []
        properties_colormap = None
        if self.fermi_surface.has_properties:
            # always calculate properties if they are present so we can determine
            # cmin and cmax. These are also be used for arrows and it is critical that
            # cmin and cmax are the same for properties and arrow color scales (even
            # if the colormap used is different)
            norm = self.fermi_surface.properties_ndim == 2
            properties = self.fermi_surface.all_properties(
                spins=spin, projection_axis=projection_axis, norm=norm
            )
            if isinstance(color_properties, str):
                properties_colormap = get_cmap(color_properties)
            else:
                properties_colormap = get_cmap(COLORMAP)
            cmin, cmax = _get_properties_limits(properties, cmin, cmax)

        if not color_properties or not self.fermi_surface.has_properties:
            colors = get_isosurface_colors(colors, self.fermi_surface, spin)
            properties = []
            cmin = None
            cmax = None

        arrows = []
        arrow_colormap = None
        if vector_properties and self.fermi_surface.has_properties:
            arrows = get_face_arrows(
                self.fermi_surface, spin, vector_spacing, vnorm, projection_axis
            )
            if isinstance(vector_properties, str):
                arrow_colormap = get_cmap(vector_properties)
            else:
                arrow_colormap = get_cmap(COLORMAP)

        return _FermiSurfacePlotData(
            isosurfaces=isosurfaces,
            azimuth=azimuth,
            elevation=elevation,
            colors=colors,
            properties=properties,
            arrows=arrows,
            properties_colormap=properties_colormap,
            arrow_colormap=arrow_colormap,
            cmin=cmin,
            cmax=cmax,
            hide_labels=hide_labels,
            hide_cell=hide_cell,
        )

    def _get_matplotlib_plot(
        self,
        plot_data: _FermiSurfacePlotData,
        ax: Optional[Any] = None,
        trisurf_kwargs: Optional[Dict[str, Any]] = None,
        cbar_kwargs: Optional[Dict[str, Any]] = None,
        quiver_kwargs: Optional[Dict[str, Any]] = None,
        bz_kwargs: Optional[Dict[str, Any]] = None,
        sym_pt_kwargs: Optional[Dict[str, Any]] = None,
        sym_label_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Plot the Fermi surface using matplotlib.

        Args:
            plot_data: The plot data.
            ax: Matplotlib 3D axes on which to plot.
            trisurf_kwargs: Optional arguments that are passed to ``ax.trisurf`` and
                are used to style the iso-surface.
            cbar_kwargs: Optional arguments that are passed to ``fig.colorbar``.
            quiver_kwargs: Optional arguments that are passed to ``ax.quiver`` and are
                used to style the arrows.
            bz_kwargs: Optional arguments that passed to ``Line3DCollection`` and used
                to style the Brillouin zone boundary.
            sym_pt_kwargs: Optional arguments that are passed to ``ax.scatter``
                and are used to style the high-symmetry k-point symbols.
            sym_label_kwargs: Optional arguments that are passed to ``ax.text`` and are
                used to style the high-symmetry k-point labels.

        Returns:
            matplotlib pyplot object.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        trisurf_kwargs = trisurf_kwargs or {}
        cbar_kwargs = cbar_kwargs or {}
        quiver_kwargs = quiver_kwargs or {}
        bz_kwargs = bz_kwargs or {}
        sym_pt_kwargs = sym_pt_kwargs or {}
        sym_label_kwargs = sym_label_kwargs or {}

        if ax is None:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection="3d", proj_type="persp")
        else:
            fig = plt.gcf()

        if plot_data.properties:
            polyc = None
            for (verts, faces), proj in zip(
                plot_data.isosurfaces, plot_data.properties
            ):
                x, y, z = verts.T
                polyc = ax.plot_trisurf(
                    x, y, faces, z, cmap=plot_data.properties_colormap, **trisurf_kwargs
                )
                polyc.set_array(proj)
                polyc.set_clim(plot_data.cmin, plot_data.cmax)
            if polyc:
                _mpl_cbar_style.update(cbar_kwargs)
                fig.colorbar(polyc, ax=ax, shrink=0.5, **_mpl_cbar_style)
        else:
            for c, (verts, faces) in zip(plot_data.colors, plot_data.isosurfaces):
                x, y, z = verts.T
                ax.plot_trisurf(x, y, faces, z, facecolor=c, **trisurf_kwargs)

        if plot_data.arrows is not None:
            norm = Normalize(vmin=plot_data.cmin, vmax=plot_data.cmax)
            for starts, stops, intensities in plot_data.arrows:
                colors = plot_data.arrow_colormap(norm(intensities))
                vectors = stops - starts
                for (x, y, z), (u, v, w), color in zip(starts, vectors, colors):
                    ax.quiver(x, y, z, u, v, w, color=color, **quiver_kwargs)

        # add the cell outline to the plot
        _mpl_bz_style.update(bz_kwargs)
        lines = Line3DCollection(self.reciprocal_space.lines, **_mpl_bz_style)
        ax.add_collection3d(lines)

        if not plot_data.hide_labels:
            for coords, label in zip(*self._symmetry_pts):
                _mpl_sym_pt_style.update(sym_pt_kwargs)
                _mpl_sym_label_style.update(sym_label_kwargs)
                ax.scatter(*coords, **_mpl_sym_pt_style)
                ax.text(*coords, "${}$".format(label), **_mpl_sym_label_style)

        xlim, ylim, zlim = np.linalg.norm(self.rlat, axis=1) / 2
        ax.set(xlim=(-xlim, xlim), ylim=(-ylim, ylim), zlim=(-zlim, zlim))
        ax.view_init(elev=plot_data.elevation, azim=plot_data.azimuth)
        ax.axis("off")
        plt.tight_layout()

        return plt

    def _get_plotly_plot(
        self,
        plot_data: _FermiSurfacePlotData,
        mesh_kwargs: Optional[Dict[str, Any]] = None,
        arrow_line_kwargs: Optional[Dict[str, Any]] = None,
        arrow_cone_kwargs: Optional[Dict[str, Any]] = None,
        bz_kwargs: Optional[Dict[str, Any]] = None,
        sym_pt_kwargs: Optional[Dict[str, Any]] = None,
        sym_label_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Plot the Fermi surface using plotly.

        Args:
            plot_data: The data to plot.
            mesh_kwargs: Optional arguments that are passed to ``Mesh3d`` and
                are used to style the iso-surface.
            arrow_line_kwargs: Additional keyword arguments used to style the arrow
                shaft and that are passed to ``Scatter3d``.
            arrow_cone_kwargs: Additional keyword arguments used to style the arrow cone
                and that are passed to ``Cone``.
            bz_kwargs: Optional arguments that passed to ``Scatter3d`` and used
                to style the Brillouin zone boundary.
            sym_pt_kwargs: Optional arguments that are passed to ``Scatter3d``
                and are used to style the high-symmetry k-point symbols.
            sym_label_kwargs: Optional arguments that are used in the annotations to
                style the high-symmetry k-point labels.

        Returns:
            Plotly figure object.
        """
        import plotly.graph_objs as go

        mesh_kwargs = mesh_kwargs or {}
        arrow_line_kwargs = arrow_line_kwargs or {}
        arrow_cone_kwargs = arrow_cone_kwargs or {}
        bz_kwargs = bz_kwargs or {}
        sym_pt_kwargs = sym_pt_kwargs or {}
        sym_label_kwargs = sym_label_kwargs or {}

        if _is_notebook():
            from plotly.offline import init_notebook_mode

            init_notebook_mode(connected=True)

        meshes = []
        if plot_data.properties:
            # plot mesh with colored properties
            colors = cmap_to_plotly(plot_data.properties_colormap)
            for (verts, faces), proj in zip(
                plot_data.isosurfaces, plot_data.properties
            ):
                x, y, z = verts.T
                i, j, k = faces.T
                trace = go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                    intensity=proj,
                    colorscale=colors,
                    intensitymode="cell",
                    cmin=plot_data.cmin,
                    cmax=plot_data.cmax,
                    **mesh_kwargs,
                    colorbar=_plotly_cbar_style,
                )
                meshes.append(trace)
        else:
            for c, (verts, faces) in zip(plot_data.colors, plot_data.isosurfaces):
                c = rgb_to_plotly(c)
                x, y, z = verts.T
                i, j, k = faces.T
                trace = go.Mesh3d(
                    x=x, y=y, z=z, color=c, opacity=1, i=i, j=j, k=k, **mesh_kwargs
                )
                meshes.append(trace)

        # add arrows
        if plot_data.arrows is not None:
            norm = Normalize(vmin=plot_data.cmin, vmax=plot_data.cmax)
            for starts, ends, intensities in plot_data.arrows:
                arrow_colors = plot_data.arrow_colormap(norm(intensities))
                for start, end, color in zip(starts, ends, arrow_colors):
                    arrow = plotly_arrow(
                        start,
                        end,
                        color[:3],
                        line_kwargs=arrow_line_kwargs,
                        cone_kwargs=arrow_cone_kwargs,
                    )
                    meshes.extend(arrow)

        # add the cell outline to the plot
        for line in self.reciprocal_space.lines:
            x, y, z = line.T
            _plotly_bz_style.update(bz_kwargs)
            trace = go.Scatter3d(x=x, y=y, z=z, mode="lines", **_plotly_bz_style)
            meshes.append(trace)

        scene = _plotly_scene.copy()
        if not plot_data.hide_labels:
            # plot high symmetry k-point markers
            labels = ["${}$".format(i) for i in self._symmetry_pts[1]]
            x, y, z = self._symmetry_pts[0].T
            _plotly_sym_pt_style.update(sym_pt_kwargs)
            trace = go.Scatter3d(x=x, y=y, z=z, mode="markers", **_plotly_bz_style)
            meshes.append(trace)

            # add high symmetry label
            annotations = []
            for label, (x, y, z) in zip(labels, self._symmetry_pts[0]):
                _plotly_sym_label_style.update(sym_label_kwargs)
                annotations.append(
                    dict(x=x, y=y, z=z, text=label, **_plotly_sym_label_style)
                )
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
    def _get_mayavi_plot(self, plot_data: _FermiSurfacePlotData):
        """
        Plot the Fermi surface using mayavi.

        Args:
            plot_data: The data to plot.

        Returns:
            mlab figure object.
        """
        from mlabtex import mlabtex

        mlab.figure(figure=None, bgcolor=(1, 1, 1), size=(800, 800), fgcolor=(0, 0, 0))

        if plot_data.properties:
            cmap = cmap_to_mayavi(plot_data.properties_colormap)
            for (verts, faces), proj in zip(
                plot_data.isosurfaces, plot_data.properties
            ):
                from tvtk.api import tvtk

                polydata = tvtk.PolyData(points=verts, polys=faces)
                polydata.cell_data.scalars = proj
                polydata.cell_data.scalars.name = "celldata"
                mesh = mlab.pipeline.surface(
                    polydata, vmin=plot_data.cmin, vmax=plot_data.cmax, opacity=0.8
                )
                mesh.module_manager.scalar_lut_manager.lut.table = cmap
                cb = mlab.colorbar(object=mesh, orientation="vertical", nb_labels=5)
                cb.label_text_property.bold = 0
                cb.label_text_property.italic = 0
        else:
            for c, (verts, faces) in zip(plot_data.colors, plot_data.isosurfaces):
                x, y, z = verts.T
                mlab.triangular_mesh(x, y, z, faces, color=tuple(c), opacity=0.8)

        if plot_data.arrows is not None:
            cmap = cmap_to_mayavi(plot_data.arrow_colormap)
            for starts, stops, intensities in plot_data.arrows:
                centers = (stops + starts) / 2
                vectors = stops - starts
                x, y, z = (centers - (vectors * 0.8)).T  # leave room for arrow tip
                u, v, w = vectors.T
                pnts = mlab.quiver3d(
                    x,
                    y,
                    z,
                    u,
                    v,
                    w,
                    line_width=4.5,
                    mode="arrow",
                    resolution=25,
                    scale_mode="vector",
                    scale_factor=2,
                    scalars=intensities,
                    vmin=plot_data.cmin,
                    vmax=plot_data.cmax,
                )
                pnts.module_manager.scalar_lut_manager.lut.table = cmap
                pnts.glyph.color_mode = "color_by_scalar"
                pnts.glyph.glyph_source.glyph_source.shaft_radius = 0.035
                pnts.glyph.glyph_source.glyph_source.tip_length = 0.3

        for line in self.reciprocal_space.lines:
            x, y, z = line.T
            mlab.plot3d(x, y, z, **_mayavi_rs_style)

        if not plot_data.hide_labels:
            # latexify labels
            labels = ["${}$".format(i) for i in self._symmetry_pts[1]]
            for coords, label in zip(self._symmetry_pts[0], labels):
                mlabtex(*coords, label, **_mayavi_sym_label_style)

        mlab.gcf().scene._lift()  # required to be able to set view
        mlab.view(
            azimuth=plot_data.azimuth - 180,
            elevation=plot_data.elevation - 90,
            distance="auto",
        )

        return mlab

    @requires(
        crystal_toolkit,
        "crystal_toolkit option requires crystal_toolkit to be installed.",
    )
    def _get_crystal_toolkit_plot(
        self, plot_data: _FermiSurfacePlotData, opacity: float = 1.0
    ) -> "Scene":
        """
        Get a crystal toolkit Scene showing the Fermi surface.

        The Scene can be displayed in an interactive web app using Crystal Toolkit, can
        be shown interactively in Jupyter Lab using the crystal-toolkit lab extension,
        or can be converted to JSON to store for future use.

        Args:
            plot_data: The data to plot.
            opacity: Opacity of surface. Note that due to limitations of WebGL,
                overlapping semi-transparent surfaces might result in visual artefacts.

        Returns:
            Crystal-toolkit scene.
        """
        from crystal_toolkit.core.scene import Lines, Scene, Spheres, Surface

        if plot_data.properties is not None or plot_data.arrows is not None:
            warnings.warn("crystal_toolkit plot does not support properties or arrows")

        # The implementation here is very similar to the plotly implementation, except
        # the crystal toolkit scene is constructed using the scene primitives from
        # crystal toolkit (Spheres, Surface, Lines, etc.)
        scene_contents = []

        # create a mesh for each electron band which has an isosurfaces at the Fermi
        # energy mesh data is generated by a marching cubes algorithm when the
        # FermiSurface object is created.
        surfaces = []
        for c, (verts, faces) in zip(plot_data.colors, plot_data.isosurfaces):
            c = rgb_to_plotly(c)
            positions = verts[faces].reshape(-1, 3).tolist()
            surface = Surface(positions=positions, color=c, opacity=opacity)
            surfaces.append(surface)
        fermi_surface = Scene("fermi_object", contents=surfaces)
        scene_contents.append(fermi_surface)

        # add the cell outline to the plot
        lines = Lines(positions=list(self.reciprocal_space.lines.flatten()))
        # alternatively,
        # cylinders have finite width and are lighted, but no strong reason to choose
        # one over the other
        # cylinders = Cylinders(positionPairs=self.reciprocal_space.lines.tolist(),
        #                       radius=0.01, color="rgb(0,0,0)")
        scene_contents.append(lines)

        if not plot_data.hide_labels:
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


class FermiSlicePlotter:
    """Class to plot 2D isolines through a FermiSurface.

    Args:
        fermi_slice: A slice through a Fermi surface.
        symprec: The symmetry precision in Angstrom for determining the high
            symmetry k-point labels.
    """

    def __init__(self, fermi_slice: FermiSlice, symprec: float = SYMPREC):
        self.fermi_slice = fermi_slice
        self.reciprocal_slice = fermi_slice.reciprocal_slice
        self._symmetry_pts = self.get_symmetry_points(fermi_slice, symprec=symprec)

    @staticmethod
    def get_symmetry_points(
        fermi_slice: FermiSlice, symprec: float = SYMPREC
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get the high symmetry k-points and labels for the Fermi slice.

        Args:
            fermi_slice: A fermi slice.
            symprec: The symmetry precision in Angstrom.

        Returns:
            The high symmetry k-points and labels for points that lie on the slice.
        """
        from pymatgen.symmetry.bandstructure import HighSymmKpath
        from trimesh import transform_points

        from ifermi.brillouin_zone import WignerSeitzCell
        from ifermi.kpoints import kpoints_to_first_bz

        hskp = HighSymmKpath(fermi_slice.structure, symprec=symprec)
        labels, kpoints = list(zip(*hskp.kpath["kpoints"].items()))

        if not np.allclose(
            hskp.prim.lattice.matrix, fermi_slice.structure.lattice.matrix, 1e-5
        ):
            warnings.warn("Structure does not match expected primitive cell")

        if not isinstance(
            fermi_slice.reciprocal_slice.reciprocal_space, WignerSeitzCell
        ):
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
        ax: Optional[Any] = None,
        spin: Optional[Spin] = None,
        colors: Optional[Union[str, dict, list]] = None,
        color_properties: Union[str, bool] = True,
        vector_properties: Union[str, bool] = False,
        projection_axis: Optional[Tuple[int, int, int]] = None,
        scale_linewidth: Union[bool, float] = False,
        vector_spacing: float = VECTOR_SPACING,
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
        vnorm: Optional[float] = None,
        hide_slice: bool = False,
        hide_labels: bool = False,
        hide_cell: bool = False,
        arrow_pivot: str = "tail",
        slice_kwargs: Optional[Dict[str, Any]] = None,
        cbar_kwargs: Optional[Dict[str, Any]] = None,
        quiver_kwargs: Optional[Dict[str, Any]] = None,
        bz_kwargs: Optional[Dict[str, Any]] = None,
        sym_pt_kwargs: Optional[Dict[str, Any]] = None,
        sym_label_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Plot the Fermi slice.

        Args:
            ax: Matplotlib axes object on which to plot.
            spin: Which spin channel to plot. By default plot both spin channels if
                available.
            colors: The color specification for the iso-surfaces. Valid options are:

                - A single color to use for all Fermi isolines, specified as a tuple of
                  rgb values from 0 to 1. E.g., red would be ``(1, 0, 0)``.
                - A list of colors, specified as above.
                - A dictionary of ``{Spin.up: color1, Spin.down: color2}``, where the
                  colors are specified as above.
                - A string specifying which matplotlib colormap to use. See
                  https://matplotlib.org/tutorials/colors/colormaps.html for more
                  information.
                - ``None``, in which case the default colors will be used.

            color_properties: Whether to use the properties to color the Fermi isolines.
                If the properties is a vector then the norm of the properties will be
                used. Note, this will only take effect if the Fermi slice has
                properties. If set to True, the viridis colormap will be used.
                Alternative colormaps can be selected by setting ``color_properties``
                to a matplotlib colormap name. This setting will override the ``colors``
                option. For vector properties, the arrows are colored according to the
                norm of the properties by default. If used in combination with the
                ``projection_axis`` option, the color will be determined by the dot
                product of the properties with the projection axis.
            vector_properties: Whether to plot arrows for vector properties. Note, this
                will only take effect if the Fermi slice has vector properties. If
                set to True, the viridis colormap will be used. Alternative colormaps
                can be selected by setting ``vector_properties`` to a matplotlib
                colormap name. By default, the arrows are colored according to the norm
                of the properties. If used in combination with the ``projection_axis``
                option, the color will be determined by the dot product of the
                properties with the projection axis.
            projection_axis: Projection axis that can be used to calculate the color of
                vector properties. If None, the norm of the properties will be used,
                otherwise the color will be determined by the dot product of the
                properties with the projection axis. Only has an effect when used with
                the ``vector_properties`` option.
            scale_linewidth: Scale the linewidth by the absolute value of the
                segment properties. Can be true, false or a number. If a number, then
                this will be used as the max linewidth for scaling.
            vector_spacing: The rough spacing between arrows. Uses a custom algorithm
                for resampling the Fermi surface to ensure that arrows are not too close
                together. Only has an effect when used with the ``vector_properties``
                option.
            cmin: Minimum intensity for normalising properties colors (including
                vector colors). Only has an effect when used with
                ``color_properties`` or ``vector_properties`` options.
            cmax: Maximum intensity for normalising properties colors (including
                vector colors). Only has an effect when used with
                ``color_properties`` or ``vector_properties`` options.
            vnorm: The value by which to normalize the vector lengths. For example,
                spin properties should typically have a norm of 1 whereas group
                velocity properties can have larger or smaller norms depending on the
                structure. By changing this number, the size of the vectors will be
                scaled. Note that the properties of two materials can only be compared
                quantitatively if a fixed values is used for both plots. Only has an
                effect when used with the ``vector_properties`` option.
            hide_slice: Whether to hide the Fermi surface. Only recommended in
                combination with the ``vector_properties`` option.
            hide_labels: Whether to show the high-symmetry k-point labels.
            hide_cell: Whether to show the reciprocal cell boundary.
            arrow_pivot: The part of the arrow that is anchored to the X, Y grid.
                The arrow rotates about this point, options are: tail, middle, tip.
            slice_kwargs: Optional arguments that are passed to ``LineCollection`` and
                are used to style the iso slice.
            cbar_kwargs: Optional arguments that are passed to ``fig.colorbar``.
            quiver_kwargs: Optional arguments that are passed to ``ax.quiver`` and are
                used to style the arrows.
            bz_kwargs: Optional arguments that passed to ``LineCollection`` and used
                to style the Brillouin zone boundary.
            sym_pt_kwargs: Optional arguments that are passed to ``ax.scatter``
                and are used to style the high-symmetry k-point symbols.
            sym_label_kwargs: Optional arguments that are passed to ``ax.text`` and are
                used to style the high-symmetry k-point labels.

        Returns:
            matplotlib pyplot object.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.transforms import ScaledTranslation

        slice_kwargs = slice_kwargs or {}
        cbar_kwargs = cbar_kwargs or {}
        quiver_kwargs = quiver_kwargs or {}
        bz_kwargs = bz_kwargs or {}
        sym_pt_kwargs = sym_pt_kwargs or {}
        sym_label_kwargs = sym_label_kwargs or {}

        plot_data = self._get_plot_data(
            spin=spin,
            colors=colors,
            color_properties=color_properties,
            vector_properties=vector_properties,
            projection_axis=projection_axis,
            vector_spacing=vector_spacing,
            cmin=cmin,
            cmax=cmax,
            vnorm=vnorm,
            hide_slice=hide_slice,
            hide_labels=hide_labels,
            hide_cell=hide_cell,
        )

        if ax is None:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
        else:
            fig = plt.gcf()

        # get rotation matrix that will align the longest slice length along the x-axis
        rotation = _get_rotation(self.fermi_slice.reciprocal_slice)

        if plot_data.properties:
            norm = Normalize(vmin=plot_data.cmin, vmax=plot_data.cmax)
            reference = max(abs(plot_data.cmax), abs(plot_data.cmin))

            lines = None
            for segments, proj in zip(plot_data.slices, plot_data.properties):
                if scale_linewidth is False:
                    linewidth = 2
                else:
                    if isinstance(scale_linewidth, (float, int)):
                        base_width = 4
                    else:
                        base_width = 4
                    linewidth = abs(proj) * base_width / reference

                slice_style = {"antialiaseds": True, "linewidth": linewidth}
                slice_style.update(slice_kwargs)
                lines = LineCollection(
                    np.dot(segments, rotation),
                    cmap=plot_data.properties_colormap,
                    norm=norm,
                    **slice_style,
                )
                lines.set_array(proj)  # set the values used for color mapping
                ax.add_collection(lines)
            if lines:
                _mpl_cbar_style.update(cbar_kwargs)
                fig.colorbar(lines, ax=ax, **_mpl_cbar_style)

        else:
            slice_style = {"antialiasted": True, "linewidth": 2}
            slice_style.update(slice_kwargs)
            for c, segments in zip(plot_data.colors, plot_data.slices):
                lines = LineCollection(
                    np.dot(segments, rotation), colors=c, **slice_kwargs
                )
                ax.add_collection(lines)

        if not plot_data.hide_cell:
            # add the cell outline to the plot
            rotated_lines = np.dot(self.reciprocal_slice.lines, rotation)
            _mpl_bz_style.update(bz_kwargs)
            lines = LineCollection(rotated_lines, **_mpl_bz_style)
            ax.add_collection(lines)

        if not plot_data.hide_labels:
            # shift labels a few pixels away from the high-sym points
            offset = ScaledTranslation(4 / 72, 4 / 72, fig.dpi_scale_trans)
            for coords, label in zip(*self._symmetry_pts):
                coords = np.dot(coords, rotation)
                _mpl_sym_pt_style.update(sym_pt_kwargs)
                _mpl_sym_label_style.update(sym_label_kwargs)
                ax.scatter(*coords, **_mpl_sym_pt_style)
                ax.text(
                    *coords,
                    "${}$".format(label),
                    **_mpl_sym_label_style,
                    transform=ax.transData + offset,
                )

        if plot_data.arrows is not None:
            norm = Normalize(vmin=plot_data.cmin, vmax=plot_data.cmax)
            _mpl_arrow_style["pivot"] = arrow_pivot
            _mpl_arrow_style.update(quiver_kwargs)
            for starts, stops, intensities in plot_data.arrows:
                colors = plot_data.arrow_colormap(norm(intensities))
                starts = np.dot(starts, rotation)
                stops = np.dot(stops, rotation)
                u, v = (stops - starts).T
                x, y = starts.T
                ax.quiver(x, y, u, v, color=colors, **_mpl_arrow_style)

            ax.margins(y=0.1, x=0.1)

        ax.autoscale_view()
        ax.axis("equal")
        ax.axis("off")

        return plt

    def _get_plot_data(
        self,
        spin: Optional[Spin] = None,
        colors: Optional[Union[str, dict, list]] = None,
        color_properties: Union[str, bool] = True,
        vector_properties: Union[str, bool] = False,
        projection_axis: Optional[Tuple[int, int, int]] = None,
        vector_spacing: float = VECTOR_SPACING,
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
        vnorm: Optional[float] = None,
        hide_slice: bool = False,
        hide_labels: bool = False,
        hide_cell: bool = False,
    ) -> _FermiSlicePlotData:
        """
        Get the the Fermi slice plot data.

        See ``FermiSlicePlotter.get_plot()`` for more details.

        Returns:
            The Fermi slice plot data.
        """
        from matplotlib.cm import get_cmap

        if not spin:
            spin = self.fermi_slice.spins
        elif isinstance(spin, Spin):
            spin = [spin]

        slices = []
        if not hide_slice:
            slices = self.fermi_slice.all_lines(spins=spin)

        properties = []
        properties_colormap = None
        if self.fermi_slice.has_properties:
            # always calculate properties if they are present so we can determine
            # cmin and cmax. These are also be used for arrows and it is critical that
            # cmin and cmax are the same for properties and arrow color scales (even
            # if the colormap used is different)
            norm = self.fermi_slice.properties_ndim == 2
            properties = self.fermi_slice.all_properties(
                spins=spin, projection_axis=projection_axis, norm=norm
            )
            if isinstance(color_properties, str):
                properties_colormap = get_cmap(color_properties)
            else:
                properties_colormap = get_cmap(COLORMAP)
            cmin, cmax = _get_properties_limits(properties, cmin, cmax)

        if not color_properties or not self.fermi_slice.has_properties:
            colors = get_isosurface_colors(colors, self.fermi_slice, spin)
            properties = []
            cmin = None
            cmax = None

        arrows = []
        arrow_colormap = None
        if vector_properties and self.fermi_slice.has_properties:
            arrows = get_segment_arrows(
                self.fermi_slice, spin, vector_spacing, vnorm, projection_axis
            )
            if isinstance(vector_properties, str):
                arrow_colormap = get_cmap(vector_properties)
            else:
                arrow_colormap = get_cmap(COLORMAP)

        return _FermiSlicePlotData(
            slices=slices,
            colors=colors,
            properties=properties,
            arrows=arrows,
            properties_colormap=properties_colormap,
            arrow_colormap=arrow_colormap,
            cmin=cmin,
            cmax=cmax,
            hide_labels=hide_labels,
            hide_cell=hide_cell,
        )


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


def save_plot(plot: Any, filename: Union[Path, str], scale: float = SCALE):
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
        if "html" in filename:
            from plotly.offline import plot as show_plotly

            show_plotly(plot, include_mathjax="cdn", filename=filename, auto_open=False)
        else:
            if kaleido is None:
                raise ValueError(
                    "kaleido package required to save static ploty images\n"
                    "please install it using:\npip install kaleido"
                )
            plot.write_image(
                filename, engine="kaleido", scale=scale, width=750, height=750
            )
    elif plot_type == "mayavi":
        plot.savefig(filename, magnification=scale)


def get_plot_type(plot: Any) -> str:
    """Get the plot type.

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
    fermi_object: Union[FermiSurface, FermiSlice],
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
        fermi_object: A Fermi surface or Fermi slice object.
        spins: A list of spins for which colors will be generated.

    Returns:
        The colors as a list of tuples, where each color is specified as the rgb values
        from 0 to 1. E.g., red would be ``(1, 0, 0)``.
    """
    from matplotlib.cm import get_cmap

    if isinstance(fermi_object, FermiSurface):
        n_objects_per_band = fermi_object.n_surfaces_per_band
    else:
        n_objects_per_band = fermi_object.n_lines_per_band

    surface_multiplicity = []
    for spin in spins:
        for band_idx in sorted(list(n_objects_per_band[spin].keys())):
            surface_multiplicity.append(n_objects_per_band[spin][band_idx])

    n_objects = len(surface_multiplicity)

    if isinstance(colors, (tuple, list, np.ndarray)):
        if isinstance(colors[0], (tuple, list, np.ndarray)):
            # colors is a list of colors
            cc = list(colors) * (len(colors) // n_objects + 1)
            color_list = cc[:n_objects]
        else:
            # colors is a single color specification
            color_list = [colors] * n_objects

    elif isinstance(colors, dict):
        if len(colors) < len(spins):
            raise ValueError(
                "colors dict must have same number of spin channels as spins to plot"
            )
        return [
            colors[s] for s in spins for _ in range(fermi_object.n_surfaces_per_spin[s])
        ]

    elif isinstance(colors, str):
        # get rid of alpha channel
        color_list = [i[:3] for i in get_cmap(colors)(np.linspace(0, 1, n_objects))]

    else:
        from plotly.colors import qualitative, unconvert_from_RGB_255, unlabel_rgb

        cc = qualitative.Prism * (len(qualitative.Prism) // n_objects + 1)
        color_list = [unconvert_from_RGB_255(unlabel_rgb(c)) for c in cc[:n_objects]]

    return [c for c, n in zip(color_list, surface_multiplicity) for _ in range(n)]


def get_face_arrows(
    fermi_surface: FermiSurface,
    spins: List[Spin],
    vector_spacing: float,
    vnorm: Optional[float],
    projection_axis: Optional[Tuple[int, int, int]],
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Get face arrows from vector properties.

    Args:
        fermi_surface: The fermi surface containing the isosurfaces and properties.
        spins: Spin channels from which to extract arrows.
        vector_spacing: The rough spacing between arrows. Uses a custom algorithm for
            resampling the Fermi surface to ensure that arrows are not too close
            together.
        vnorm: The value by which to normalize the vector lengths. For example,
            spin properties should typically have a norm of 1 whereas group velocity
            properties can have larger or smaller norms depending on the structure.
            By changing this number, the size of the vectors will be scaled. Note that
            the properties of two materials can only be compared quantitatively if a
            fixed values is used for both plots.
        projection_axis: Projection axis that can be used to calculate the color of
            vector projections. If None, the norm of the properties will be used,
            otherwise the color will be determined by the dot product of the properties
            with the projection axis.

    Returns:
        The arrows, as a list of (starts, stops, intensities) for each face. The
        starts and stops are numpy arrays with the shape (narrows, 3) and intensities
        is a numpy array with the shape (narrows, ). The intensities are used
        to color the arrows during plotting.
    """
    centers = []
    intensity = []
    vectors = []
    for spin in spins:
        for isosurface in fermi_surface.isosurfaces[spin]:
            if isosurface.properties_ndim != 2:
                continue

            face_idx = isosurface.sample_uniform(vector_spacing)

            # get the center of each of face in cartesian coords
            faces = isosurface.faces[face_idx]
            centers.append(isosurface.vertices[faces].mean(axis=1))

            vectors.append(isosurface.properties[face_idx])
            if projection_axis is None:
                # intensity is the norm of the properties
                intensities = isosurface.properties_norms[face_idx]
            else:
                # get intensity from projection of the vector onto axis
                intensities = isosurface.scalar_projection(projection_axis)[face_idx]
            intensity.append(intensities)

    if vnorm is None:
        property_norms = fermi_surface.all_properties(spins=spins, norm=True)
        vnorm = np.max([np.max(x) for x in property_norms])

    arrows = []
    for face_vectors, face_centers, face_intensity in zip(vectors, centers, intensity):
        face_vectors *= 0.14 / vnorm  # 0.14 is magic scaling factor for vector length
        start = face_centers - face_vectors / 2
        stop = start + face_vectors
        arrows.append((start, stop, face_intensity))

    return arrows


def get_segment_arrows(
    fermi_slice: FermiSlice,
    spins: Collection[Spin],
    vector_spacing: float,
    vnorm: Optional[float],
    projection_axis: Optional[Tuple[int, int, int]],
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Get segment arrows from vector properties.

    Args:
        fermi_slice: The Fermi slice containing the isolines and properties.
        spins: Spin channels from which to extract arrows.
        vector_spacing: The rough spacing between arrows. Uses a custom algorithm for
            resampling the Fermi slic to ensure that arrows are not too close
            together.
        vnorm: The value by which to normalize the vector lengths. For example,
            spin properties should typically have a norm of 1 whereas group velocity
            properties can have larger or smaller norms depending on the structure.
            By changing this number, the size of the vectors will be scaled. Note that
            the properties of two materials can only be compared quantitatively if a
            fixed values is used for both plots.
        projection_axis: Projection axis that can be used to calculate the color of
            vector projects. If None, the norm of the properties will be used,
            otherwise the color will be determined by the dot product of the properties
            with the properties axis.

    Returns:
        The arrows, as a list of (starts, stops, intensities) for each face. The
        starts and stops are numpy arrays with the shape (narrows, 3) and intensities
        is a numpy array with the shape (narrows, ). The intensities are used
        to color the arrows during plotting.
    """
    from trimesh import transform_points

    centers = []
    intensity = []
    vectors = []
    for spin in spins:
        for isoline in fermi_slice.isolines[spin]:
            if isoline.properties_ndim != 2:
                continue

            segment_idx = isoline.sample_uniform(vector_spacing)

            # get the center of each of segment in cartesian coords
            centers.append(isoline.segments[segment_idx].mean(axis=1))

            vectors.append(isoline.properties[segment_idx])
            if projection_axis is None:
                # properties intensity is the norm of the properties
                intensities = isoline.properties_norms[segment_idx]
            else:
                # get properties intensity from properties of the vector onto axis
                intensities = isoline.scalar_projection(projection_axis)[segment_idx]

            intensity.append(intensities)

    if vnorm is None:
        property_norms = fermi_slice.all_properties(spins=spins, norm=True)
        vnorm = np.max([np.max(x) for x in property_norms])

    arrows = []
    for segment_vectors, segment_centers, segment_intensity in zip(
        vectors, centers, intensity
    ):
        segment_vectors *= 0.31 / vnorm  # magic scaling factor for length

        # transform vectors onto 2D plane
        segment_vectors = transform_points(
            segment_vectors, fermi_slice.reciprocal_slice.transformation
        )[:, :2]
        start = segment_centers
        stop = start + segment_vectors
        arrows.append((start, stop, segment_intensity))

    return arrows


def _get_properties_limits(
    projections: List[np.ndarray], cmin: Optional[float], cmax: Optional[float]
) -> Tuple[float, float]:
    """
    Get the min and max properties if they are not already set.

    Args:
        projections: The properties for each Fermi surface as a list of numpy arrays.
        cmin: A minimum value that overrides the one extracted from the properties.
        cmax: A maximum value that overrides the one extracted from the properties.

    Returns:
        The projection limits as a tuple of (min, max).
    """
    if cmax is None:
        cmax = np.max([np.max(x) for x in projections])

    if cmin is None:
        cmin = np.min([np.min(x) for x in projections])

    return cmin, cmax


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


def plotly_arrow(
    start: np.ndarray,
    stop: np.ndarray,
    color: Tuple[float, float, float],
    line_kwargs: Optional[Dict[str, Any]] = None,
    cone_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    """
    Create an arrow object.

    Args:
        start: The starting coordinates.
        stop: The ending coordinates.
        color: The arrow color in rgb format as a tuple of floats from 0 to 1.
        line_kwargs: Additional keyword arguments used to style the arrow shaft and that
            are passed to ``Scatter3d``.
        cone_kwargs: Additional keyword arguments used to style the arrow cone and that
            are passed to ``Cone``.

    Returns:
        The arrow, formed by a line and cone.
    """
    import plotly.graph_objs as go

    vector = (stop - start) / np.linalg.norm(stop - start)
    color = rgb_to_plotly(color)

    line_kwargs = line_kwargs or {}
    cone_kwargs = cone_kwargs or {}

    line_style = {"line": {"width": 6, "color": color}, "showlegend": False}
    line_style.update(line_kwargs)

    cone_style = {
        "showscale": False,
        "sizemode": "absolute",
        "sizeref": 0.08,  # magic cone length
        "anchor": "cm",
    }
    cone_style.update(cone_kwargs)

    line = go.Scatter3d(
        x=[start[0], stop[0]],
        y=[start[1], stop[1]],
        z=[start[2], stop[2]],
        mode="lines",
        **line_style,
    )
    cone = go.Cone(
        x=[stop[0]],
        y=[stop[1]],
        z=[stop[2]],
        u=[vector[0]],
        v=[vector[1]],
        w=[vector[2]],
        colorscale=[[0, color], [1, color]],
        **cone_style,
    )
    return line, cone


def rgb_to_plotly(color: Tuple[float, float, float]) -> str:
    """
    Get a plotly formatted color from rgb values.

    Args:
        color: The color in rgb format as a tuple of three floats from 0 to 1.

    Returns:
        The plotly formatted color.
    """
    from plotly.colors import convert_to_RGB_255, label_rgb

    return label_rgb(convert_to_RGB_255(color))


def cmap_to_plotly(colormap: Colormap) -> List[str]:
    """
    Convert a matplotlib colormap to plotly colorscale format.

    Args:
        colormap: A matplotlib colormap object.

    Returns:
        The equivalent plotly colorscale.
    """
    from plotly.colors import make_colorscale

    rgb_colors = colormap(np.linspace(0, 1, 255))[:, :3]
    return make_colorscale([rgb_to_plotly(color) for color in rgb_colors])


def cmap_to_mayavi(colormap: Colormap) -> np.ndarray:
    """
    Convert a matplotlib colormap to mayavi format.

    Args:
        colormap: A matplotlib colormap object.

    Returns:
        The equivalent mayavi colormap, as a (255, 4) numpy array.
    """
    return (colormap(np.linspace(0, 1, 255)) * 255).astype(int)


def _get_rotation(reciprocal_slice) -> np.ndarray:
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
    return rotation.T


def _is_notebook():
    """Check if running in a jupyter notebook."""
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:
            raise ImportError("console")
        if "VSCODE_PID" in os.environ:
            raise ImportError("vscode")
    except Exception:
        return False
    else:
        return True
