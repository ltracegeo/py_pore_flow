import os

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from PySide2 import QtCore, QtWidgets

os.environ.setdefault("QT_API", "pyside2")
try:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# name of the point-data array vtkGlyph3D adds when GeneratePointIds is on: it maps each
# glyph-output point back to the input point (= pore id) that generated it
INPUT_POINT_IDS = "InputPointIds"
# cell-data array we attach to the throat lines; vtkTubeFilter copies it onto every output
# tube cell, so a picked/colored tube cell maps back to its throat id
THROAT_IDS = "ThroatIds"

PORE_COLOR = (0.60, 0.72, 1.00)
THROAT_COLOR = (0.55, 0.55, 0.55)
HIGHLIGHT_COLOR = (1.00, 0.35, 0.25)

COLOR_MODE_NONE = "None"
COLOR_MODE_FILLING = "Filling"
COLOR_MODE_WATER_CONDUCTIVITY = "Water Conductivity"
COLOR_MODE_OIL_CONDUCTIVITY = "Oil Conductivity"
COLOR_MODES = [
    COLOR_MODE_NONE,
    COLOR_MODE_FILLING,
    COLOR_MODE_WATER_CONDUCTIVITY,
    COLOR_MODE_OIL_CONDUCTIVITY,
]

RED_CHANNEL = 0
BLUE_CHANNEL = 2


class DiagnosticsNetworkWidget(QtWidgets.QWidget):
    poreClicked = QtCore.Signal(int)
    throatClicked = QtCore.Signal(int)

    def __init__(self, diagnostics, parent=None):
        super().__init__(parent)
        self._diagnostics = diagnostics

        # Widgets
        network_widget = QtWidgets.QWidget()
        network_layout = QtWidgets.QVBoxLayout(network_widget)

        self.color_combo = QtWidgets.QComboBox()
        self.color_combo.addItems(COLOR_MODES)
        self.color_combo.currentTextChanged.connect(self._apply_color_mode)

        color_bar = QtWidgets.QHBoxLayout()
        color_bar.addWidget(QtWidgets.QLabel("Color:"))
        color_bar.addWidget(self.color_combo)
        color_bar.addStretch()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(network_widget)
        layout.addLayout(color_bar)

        # Vtk
        network = diagnostics.network
        self._coords = np.asarray(network["pore.coords"], dtype=float)
        self._pore_radii = np.asarray(network["pore.radius"], dtype=float)
        self._throat_conns = np.asarray(network["throat.conns"], dtype=int)
        throat_radii = np.asarray(network.get("throat.radius", []), dtype=float)
        self._throat_tube_radius = self._compute_tube_radius(throat_radii, self._pore_radii)

        self._points = self._build_points(self._coords)

        self._vtk_widget = QVTKRenderWindowInteractor(network_widget)
        network_layout.addWidget(self._vtk_widget)
        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(1.0, 1.0, 1.0)
        self._vtk_widget.GetRenderWindow().AddRenderer(self._renderer)
        self._interactor = self._vtk_widget.GetRenderWindow().GetInteractor()

        self._pore_actor = self._build_pores(self._points, self._pore_radii)
        self._renderer.AddActor(self._pore_actor)

        self._throat_actor = self._build_throats(self._points, self._throat_conns)
        if self._throat_actor is not None:
            self._renderer.AddActor(self._throat_actor)

        self._pore_highlight = self._make_pore_highlight_actor()
        self._renderer.AddActor(self._pore_highlight)
        self._throat_highlight = self._make_throat_highlight_actor()
        self._renderer.AddActor(self._throat_highlight)

        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.001)

        self._style = vtk.vtkInteractorStyleTrackballCamera()
        self._interactor.SetInteractorStyle(self._style)
        self._interactor.SetPicker(self._picker)
        self._interactor.AddObserver("LeftButtonPressEvent", self._on_left_button_press)

        self._renderer.ResetCamera()
        self._interactor.Initialize()

    @staticmethod
    def _compute_tube_radius(throat_radii, pore_radii):
        radius = float(np.median(throat_radii)) if throat_radii.size else 0.0
        if not np.isfinite(radius) or radius <= 0:
            radius = float(np.median(pore_radii)) * 0.25
        return radius

    @staticmethod
    def _build_points(coords):
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(len(coords))
        for i, (x, y, z) in enumerate(coords):
            points.SetPoint(i, x, y, z)
        return points

    def _build_pores(self, points, radii):
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)

        radius_array = vtk.vtkFloatArray()
        radius_array.SetName("radius")
        radius_array.SetNumberOfValues(len(radii))
        for i, r in enumerate(radii):
            radius_array.SetValue(i, float(r))
        polydata.GetPointData().SetScalars(radius_array)

        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(1.0)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)

        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceConnection(sphere.GetOutputPort())
        glyph.SetInputData(polydata)
        glyph.SetScaleModeToScaleByScalar()
        glyph.SetScaleFactor(1.0)
        glyph.GeneratePointIdsOn()
        glyph.Update()

        self._pore_polydata = vtk.vtkPolyData()
        self._pore_polydata.DeepCopy(glyph.GetOutput())
        self._pore_point_to_id = vtk_to_numpy(self._pore_polydata.GetPointData().GetArray(INPUT_POINT_IDS)).astype(int)

        self._pore_mapper = vtk.vtkPolyDataMapper()
        self._pore_mapper.SetInputData(self._pore_polydata)
        self._pore_mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(self._pore_mapper)
        actor.GetProperty().SetColor(*PORE_COLOR)
        return actor

    def _build_throats(self, points, throat_conns):
        self._throat_polydata = None
        self._throat_mapper = None
        self._throat_cell_to_id = None
        if len(throat_conns) == 0:
            return None

        lines = vtk.vtkCellArray()
        throat_ids = vtk.vtkIdTypeArray()
        throat_ids.SetName(THROAT_IDS)
        throat_ids.SetNumberOfComponents(1)
        for throat_id, (pore_a, pore_b) in enumerate(throat_conns):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, int(pore_a))
            line.GetPointIds().SetId(1, int(pore_b))
            lines.InsertNextCell(line)
            throat_ids.InsertNextValue(throat_id)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        polydata.GetCellData().AddArray(throat_ids)

        tube = vtk.vtkTubeFilter()
        tube.SetInputData(polydata)
        tube.SetRadius(self._throat_tube_radius)
        tube.SetNumberOfSides(6)
        tube.CappingOn()
        tube.Update()

        self._throat_polydata = vtk.vtkPolyData()
        self._throat_polydata.DeepCopy(tube.GetOutput())
        self._throat_cell_to_id = vtk_to_numpy(self._throat_polydata.GetCellData().GetArray(THROAT_IDS)).astype(int)

        self._throat_mapper = vtk.vtkPolyDataMapper()
        self._throat_mapper.SetInputData(self._throat_polydata)
        self._throat_mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(self._throat_mapper)
        actor.GetProperty().SetColor(*THROAT_COLOR)
        return actor

    def _make_pore_highlight_actor(self):
        self._pore_highlight_source = vtk.vtkSphereSource()
        self._pore_highlight_source.SetThetaResolution(20)
        self._pore_highlight_source.SetPhiResolution(20)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self._pore_highlight_source.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*HIGHLIGHT_COLOR)
        actor.GetProperty().SetOpacity(0.45)
        actor.PickableOff()
        actor.VisibilityOff()
        return actor

    def _make_throat_highlight_actor(self):
        self._throat_highlight_points = vtk.vtkPoints()
        self._throat_highlight_points.SetNumberOfPoints(2)
        self._throat_highlight_points.SetPoint(0, 0.0, 0.0, 0.0)
        self._throat_highlight_points.SetPoint(1, 0.0, 0.0, 0.0)

        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, 0)
        line.GetPointIds().SetId(1, 1)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(line)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(self._throat_highlight_points)
        polydata.SetLines(lines)

        tube = vtk.vtkTubeFilter()
        tube.SetInputData(polydata)
        tube.SetRadius(self._throat_tube_radius * 1.6)
        tube.SetNumberOfSides(10)
        tube.CappingOn()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*HIGHLIGHT_COLOR)
        actor.GetProperty().SetOpacity(0.55)
        actor.PickableOff()
        actor.VisibilityOff()
        return actor

    def refresh_colors(self):
        self._apply_color_mode(self.color_combo.currentText())

    def _apply_color_mode(self, mode):
        if mode == COLOR_MODE_FILLING:
            self._apply_element_colors(self._fill_colors)
        elif mode == COLOR_MODE_WATER_CONDUCTIVITY:
            self._apply_element_colors(self._water_conductivity_colors)
        elif mode == COLOR_MODE_OIL_CONDUCTIVITY:
            self._apply_element_colors(self._oil_conductivity_colors)
        else:
            self._pore_mapper.ScalarVisibilityOff()
            if self._throat_mapper is not None:
                self._throat_mapper.ScalarVisibilityOff()
        self._vtk_widget.GetRenderWindow().Render()

    def _apply_element_colors(self, color_func):
        network = self._diagnostics.network

        pore_colors = color_func(network, "pore")
        self._set_point_colors(self._pore_polydata, self._pore_mapper, pore_colors[self._pore_point_to_id])

        if self._throat_mapper is not None:
            throat_colors = color_func(network, "throat")
            self._set_cell_colors(self._throat_polydata, self._throat_mapper, throat_colors[self._throat_cell_to_id])

    @staticmethod
    def _fill_colors(network, element_type):
        filled = np.asarray(network[f"{element_type}.water_filled"], dtype=bool).ravel()
        ws = np.clip(np.asarray(network[f"{element_type}.water_saturation"], dtype=float).ravel(), 0.0, 1.0)

        colors = np.zeros((filled.size, 3), dtype=np.uint8)
        colors[filled, 2] = (255 * ws[filled]).astype(np.uint8)  # blue channel
        not_filled = ~filled
        colors[not_filled, 0] = (255 * (1.0 - ws[not_filled])).astype(np.uint8)  # red channel
        return colors

    @classmethod
    def _water_conductivity_colors(cls, network, element_type):
        return cls._conductivity_ratio_colors(network, element_type, "water", BLUE_CHANNEL)

    @classmethod
    def _oil_conductivity_colors(cls, network, element_type):
        return cls._conductivity_ratio_colors(network, element_type, "oil", RED_CHANNEL)

    @staticmethod
    def _conductivity_ratio_colors(network, element_type, phase, channel):
        multiphase = np.asarray(network[f"{element_type}.{phase}_multiphase_conductivity"], dtype=float).ravel()
        singlephase = np.asarray(network[f"{element_type}.{phase}_singlephase_conductivity"], dtype=float).ravel()

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(singlephase > 0, multiphase / singlephase, 0.0)
        ratio = np.clip(ratio, 0.0, 1.0)

        colors = np.zeros((ratio.size, 3), dtype=np.uint8)
        colors[:, channel] = (255 * ratio).astype(np.uint8)
        return colors

    @staticmethod
    def _to_vtk_colors(colors):
        vtk_colors = numpy_to_vtk(np.ascontiguousarray(colors), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_colors.SetName("colors")
        return vtk_colors

    def _set_point_colors(self, polydata, mapper, colors):
        polydata.GetPointData().SetScalars(self._to_vtk_colors(colors))
        mapper.SetScalarModeToUsePointData()
        mapper.SetColorModeToDirectScalars()
        mapper.ScalarVisibilityOn()

    def _set_cell_colors(self, polydata, mapper, colors):
        polydata.GetCellData().SetScalars(self._to_vtk_colors(colors))
        mapper.SetScalarModeToUseCellData()
        mapper.SetColorModeToDirectScalars()
        mapper.ScalarVisibilityOn()

    def _on_left_button_press(self, _obj, _event):
        x, y = self._interactor.GetEventPosition()
        self._picker.Pick(x, y, 0, self._renderer)
        if self._picker.GetCellId() < 0:
            return

        actor = self._picker.GetActor()
        if actor is self._pore_actor:
            self._pick_pore()
        elif actor is self._throat_actor:
            self._pick_throat()

    def _pick_pore(self):
        point_id = self._picker.GetPointId()
        if not (0 <= point_id < self._pore_point_to_id.size):
            return
        pore_id = int(self._pore_point_to_id[point_id])
        self.highlight_pore(pore_id)
        self.poreClicked.emit(pore_id)

    def _pick_throat(self):
        cell_id = self._picker.GetCellId()
        if self._throat_cell_to_id is None or not (0 <= cell_id < self._throat_cell_to_id.size):
            return
        throat_id = int(self._throat_cell_to_id[cell_id])
        self.highlight_throat(throat_id)
        self.throatClicked.emit(throat_id)

    def highlight_pore(self, pore_id):
        if not (0 <= pore_id < len(self._coords)):
            return
        x, y, z = self._coords[pore_id]
        self._pore_highlight_source.SetCenter(x, y, z)
        self._pore_highlight_source.SetRadius(self._highlight_radius(pore_id))
        self._throat_highlight.VisibilityOff()
        self._pore_highlight.VisibilityOn()
        self._vtk_widget.GetRenderWindow().Render()

    def highlight_throat(self, throat_id):
        if not (0 <= throat_id < len(self._throat_conns)):
            return
        pore_a, pore_b = self._throat_conns[throat_id]
        self._throat_highlight_points.SetPoint(0, *self._coords[int(pore_a)])
        self._throat_highlight_points.SetPoint(1, *self._coords[int(pore_b)])
        self._throat_highlight_points.Modified()
        self._pore_highlight.VisibilityOff()
        self._throat_highlight.VisibilityOn()
        self._vtk_widget.GetRenderWindow().Render()

    def _highlight_radius(self, pore_id):
        r = float(self._pore_radii[pore_id]) if pore_id < self._pore_radii.size else 0.0
        return r * 1.25 if r > 0 else float(np.median(self._pore_radii)) * 1.25
