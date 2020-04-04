"""
This module implements a plotter for the Fermi-Surface of a material
todo:
* Remap into Brillioun zone (Wigner Seitz cell)
* Get Latex working for labels
* Do projections onto arbitrary surface
* Comment more
* Think about classes/methods, maybe restructure depending on sumo layout
"""

from typing import Any, List, Optional, Tuple, Union

import colorlover as cl
import numpy as np
from matplotlib import cm
from monty.dev import requires
from monty.json import MSONable

from ifermi.brillouin_zone import ReciprocalCell
from ifermi.fermi_surface import FermiSurface
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

_plotly_high_sym_label_style = {
    "mode": "markers+text",
    "marker": {"size": 5, "color": "black"},
    "name": "Markers and Text",
    "textposition": "bottom center",
}

_mayavi_high_sym_label_style = {
    "color": (0, 0, 0),
    "scale": 0.1,
    "orientation": (90.0, 0.0, 0.0),
}

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

_mayavi_rs_style = {
    "color": (0.0, 0.0, 0.0),
    "tube_radius": 0.005,
    "representation": "surface",
}


class FSPlotter(MSONable):
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
    def get_symmetry_points(fermi_surface) -> Tuple[np.ndarray, List[str]]:
        """
        Get the high symmetry k-points and labels for the Fermi surface.

        Args:
            fermi_surface: A fermi surface.

        Returns:
            The high symmetry k-points and labels.
        """
        kpoints, labels = [], []
        hskp = HighSymmKpath(fermi_surface.structure)
        all_kpoints, all_labels = hskp.get_kpoints(coords_are_cartesian=False)

        for kpoint, label in zip(all_kpoints, all_labels):
            if not len(label) == 0:
                kpoints.append(kpoint)
                labels.append(label)

        if isinstance(fermi_surface.reciprocal_space, ReciprocalCell):
            kpoints = kpoints_to_first_bz(np.array(kpoints))

        kpoints = np.dot(kpoints, fermi_surface.reciprocal_space.reciprocal_lattice)
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
        if plot_type == "mpl":
            self.plot_matplotlib(
                filename=filename, interactive=interactive, colors=colors, **plot_kwargs
            )
        elif plot_type == "plotly":
            self.plot_plotly(
                filename=filename, interactive=interactive, colors=colors, **plot_kwargs
            )
        elif plot_type == "mayavi":
            self.plot_mayavi(
                filename=filename, interactive=interactive, colors=colors, **plot_kwargs
            )
        else:
            types = ["mpl", "plotly", "mayavi"]
            raise ValueError(
                "Plot type not recognised, valid options: {}".format(types)
            )

    def get_isosurfaces_and_colors(
        self,
        spin: Optional[Spin] = None,
        colors: Optional[Union[str, dict, list]] = None
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

        if isinstance(colors, dict):
            if len(colors) < len(spin):
                raise ValueError(
                    "colors dictionary should have the same number of spin channels as"
                    "spins to plot"
                )
            colors = []
            for s in spin:
                colors.extend([colors[s]] * len(self.fermi_surface.isosurfaces[s]))

        elif isinstance(colors, str):
            cmap = cm.get_cmap(colors)
            colors = cmap(np.linspace(0, 1, len(isosurfaces)))

        elif colors is None:
            colors = np.random.random((len(isosurfaces), 3))

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
        corners = self.reciprocal_space.faces
        lines = Line3DCollection(corners, colors="k", linewidths=bz_linewidth)
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
        for facet in self.reciprocal_space.faces:
            x, y, z = zip(*facet)
            line = dict(color="black", width=3)
            trace = go.Scatter3d(x=x, y=y, z=z, mode="lines", line=line)
            meshes.append(trace)

        # plot high symmetry labels
        labels = [i.replace(r"\Gamma", "\u0393") for i in self._symmetry_pts[1]]
        x, y, z = zip(*self._symmetry_pts[0])
        trace = go.Scatter3d(x=x, y=y, z=z, **_plotly_high_sym_label_style)
        meshes.append(trace)

        annotations = []
        for label, (x, y, z) in zip(labels, self._symmetry_pts[0]):
            # annotations always appear on top of the plot
            style = dict(xshift=10, yshift=10, text=label, showarrow=False)
            annotations.append(dict(x=x, y=y, z=z, **style))
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
            plotly.io.write_image(
                fig, str(filename), format="pdf", width=600, height=600, scale=5
            )

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

        for facet in self.reciprocal_space.faces:
            x, y, z = zip(*facet)
            mlab.plot3d(x, y, z, **_mayavi_rs_style)

        if not colors:
            colors = np.random.random((20, 3))

        for c, (verts, faces) in zip(colors, self.fermi_surface.isosurfaces):
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


class FSPlotter2D(object):
    def __init__(self, fs: FermiSurface, plane_orig, plane_norm):

        self._plane_orig = plane_orig
        self._plane_norm = plane_norm
        self._fs = fs

    def fs2d_plot_data(self, plot_type="mpl"):
        import meshcut

        plane_orig = self._plane_orig
        plane_norm = self._plane_norm

        plane = meshcut.Plane(plane_orig, plane_norm)

        if plot_type == "mayavi":

            mlab.figure(figure=None, bgcolor=(1, 1, 1), size=(800, 800))

        elif plot_type == "mpl":

            fig = plt.figure()

            ax = plt.axes(projection="3d")

        for surface in self._fs._iso_surface:

            verts = surface[0]
            faces = surface[1]

            mesh = meshcut.TriangleMesh(verts, faces)

            P = meshcut.cross_section_mesh(mesh, plane)

            for p in P:
                p = np.array(p)
                if plot_type == "mayavi":
                    mlab.plot3d(
                        p[:, 0],
                        p[:, 1],
                        p[:, 2],
                        tube_radius=None,
                        line_width=3.0,
                        color=(0.0, 0.0, 0.0),
                    )
                elif plot_type == "mpl":
                    ax.plot3D(p[:, 0], p[:, 1], p[:, 2], color="k")

        if plot_type == "mayavi":

            mlab.show()

        elif plot_type == "mpl":

            ax.set_xticks([])
            ax.set_yticks([])
            plt.show()


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

