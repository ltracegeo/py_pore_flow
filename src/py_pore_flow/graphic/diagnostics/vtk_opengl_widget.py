"""A VTK render view based on QOpenGLWidget.

It replaces 'vtkmodules.qt.QVTKRenderWindowInteractor', whose constructor calls 'winId()' and
sets 'Qt.WA_PaintOnScreen'. Those turn the view into a native window and, on Windows, promote
every ancestor to a native window as well, which stops the sibling widgets of the application
hosting the view from being painted.

A QOpenGLWidget renders into a framebuffer object that Qt composites like any other widget,
then no native window is created. The rendering is delegated to a vtkGenericOpenGLRenderWindow,
which doesn't own the OpenGL context: Qt owns it, and the render window asks for it through the
'WindowMakeCurrentEvent' event.
"""

from PySide2.QtCore import Qt, QTimer
from PySide2.QtWidgets import QOpenGLWidget, QSizePolicy

import vtk


class VTKOpenGLWidget(QOpenGLWidget):
    """Widget that renders a vtkRenderer, replacing QVTKRenderWindowInteractor.

    It provides the same 'GetRenderWindow' method, then the renderers are added the usual way.
    Rendering must be requested through this widget's 'Render' method, and not through the
    render window's one, because a QOpenGLWidget can only render inside its 'paintGL' method.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._renderWindow = vtk.vtkGenericOpenGLRenderWindow()
        # Nothing can be rendered before Qt creates the OpenGL context. The render window
        # skips every render request while it isn't ready, instead of rendering to no context.
        self._renderWindow.SetReadyForRendering(False)
        self._renderWindow.SetOwnContext(False)
        self._renderWindow.AddObserver("WindowMakeCurrentEvent", self.__onMakeCurrent)

        self._interactor = vtk.vtkGenericRenderWindowInteractor()
        self._interactor.SetRenderWindow(self._renderWindow)
        self._interactor.AddObserver("CreateTimerEvent", self.__onCreateTimer)
        self._interactor.AddObserver("DestroyTimerEvent", self.__onDestroyTimer)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._interactor.TimerEvent)

        self._activeButton = Qt.NoButton
        self._wheelDelta = 0
        self._lastX = 0
        self._lastY = 0

        self.setFocusPolicy(Qt.WheelFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def GetRenderWindow(self) -> vtk.vtkGenericOpenGLRenderWindow:
        return self._renderWindow

    def GetInteractor(self) -> vtk.vtkGenericRenderWindowInteractor:
        return self._interactor

    def Render(self) -> None:
        """Request a new frame. Qt calls 'paintGL' when the OpenGL context is current."""
        self.update()

    def Finalize(self) -> None:
        """Release the render window's OpenGL resources while the context is still alive."""
        if not self._renderWindow.GetReadyForRendering():
            return

        self.makeCurrent()
        self._renderWindow.SetReadyForRendering(False)
        self._renderWindow.Finalize()
        self.doneCurrent()

    # QOpenGLWidget's interface

    def initializeGL(self) -> None:
        super().initializeGL()

        # Bind the render window to the context Qt has just created and made current
        self._renderWindow.InitializeFromCurrentContext()
        # Qt owns the framebuffer object being rendered to, then VTK must blit to it
        self._renderWindow.SetFrameBlitModeToBlitToCurrent()
        self._renderWindow.SetReadyForRendering(True)
        self._interactor.Initialize()

    def resizeGL(self, width: int, height: int) -> None:
        super().resizeGL(width, height)

        # Qt recreates its framebuffer object when the widget is resized
        self._renderWindow.InitializeFromCurrentContext()

        ratio = self.devicePixelRatioF()
        scaledWidth = int(round(width * ratio))
        scaledHeight = int(round(height * ratio))

        self._renderWindow.SetDPI(int(round(72 * ratio)))
        self._renderWindow.SetSize(scaledWidth, scaledHeight)
        self._interactor.SetSize(scaledWidth, scaledHeight)
        self._interactor.ConfigureEvent()

    def paintGL(self) -> None:
        super().paintGL()
        self._renderWindow.Render()

    def closeEvent(self, event) -> None:
        self.Finalize()
        super().closeEvent(event)

    # Event forwarding to the interactor

    def enterEvent(self, event) -> None:
        self.__setEventInformation(self._lastX, self._lastY, event)
        self._interactor.EnterEvent()

    def leaveEvent(self, event) -> None:
        self.__setEventInformation(self._lastX, self._lastY, event)
        self._interactor.LeaveEvent()

    def mousePressEvent(self, event) -> None:
        repeat = 1 if event.type() == event.MouseButtonDblClick else 0
        self.__setEventInformation(event.x(), event.y(), event, repeat=repeat)

        self._activeButton = event.button()
        if self._activeButton == Qt.LeftButton:
            self._interactor.LeftButtonPressEvent()
        elif self._activeButton == Qt.RightButton:
            self._interactor.RightButtonPressEvent()
        elif self._activeButton == Qt.MiddleButton:
            self._interactor.MiddleButtonPressEvent()

    def mouseReleaseEvent(self, event) -> None:
        self.__setEventInformation(event.x(), event.y(), event)

        if self._activeButton == Qt.LeftButton:
            self._interactor.LeftButtonReleaseEvent()
        elif self._activeButton == Qt.RightButton:
            self._interactor.RightButtonReleaseEvent()
        elif self._activeButton == Qt.MiddleButton:
            self._interactor.MiddleButtonReleaseEvent()

        self._activeButton = Qt.NoButton

    def mouseMoveEvent(self, event) -> None:
        self._lastX = event.x()
        self._lastY = event.y()
        self.__setEventInformation(self._lastX, self._lastY, event)
        self._interactor.MouseMoveEvent()

    def wheelEvent(self, event) -> None:
        self._wheelDelta += event.angleDelta().y()

        if self._wheelDelta >= 120:
            self._interactor.MouseWheelForwardEvent()
            self._wheelDelta = 0
        elif self._wheelDelta <= -120:
            self._interactor.MouseWheelBackwardEvent()
            self._wheelDelta = 0

    def keyPressEvent(self, event) -> None:
        self.__setEventInformation(self._lastX, self._lastY, event, key=self.__getKeyChar(event))
        self._interactor.KeyPressEvent()
        self._interactor.CharEvent()

    def keyReleaseEvent(self, event) -> None:
        self.__setEventInformation(self._lastX, self._lastY, event, key=self.__getKeyChar(event))
        self._interactor.KeyReleaseEvent()

    def __setEventInformation(self, x: int, y: int, event, key: str = "\0", repeat: int = 0) -> None:
        modifiers = event.modifiers() if hasattr(event, "modifiers") else Qt.NoModifier
        control = bool(modifiers & Qt.ControlModifier)
        shift = bool(modifiers & Qt.ShiftModifier)

        # VTK's origin is at the bottom left corner, Qt's one is at the top left corner
        ratio = self.devicePixelRatioF()
        self._interactor.SetEventInformation(
            int(round(x * ratio)),
            int(round((self.height() - y - 1) * ratio)),
            control,
            shift,
            key,
            repeat,
            None,
        )

    @staticmethod
    def __getKeyChar(event) -> str:
        text = event.text()
        return text[0] if text else "\0"

    def __onMakeCurrent(self, caller, event) -> None:
        self.makeCurrent()

    def __onCreateTimer(self, caller, event) -> None:
        self._timer.start(10)

    def __onDestroyTimer(self, caller, event) -> None:
        self._timer.stop()
