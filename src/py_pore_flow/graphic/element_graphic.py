import math

import matplotlib.pyplot as plt
import numpy as np

WATER = 0
OIL = 1


class ElementPlotter:
    def __init__(self):
        self.oil_color = "#ffbfbf"
        self.water_color = "#abcbff"
        self.line_color = "#FFFFFF"

        self.element_graphic_list = []
        self.axis_list = []

    def add_element_graphic(self, element_graphic):
        self.element_graphic_list.append(element_graphic)

    def show(self, pause=0, block=False):
        number_of_plots = len(self.element_graphic_list)
        if number_of_plots != len(self.axis_list):
            self.axis_list = self.reset_axes(number_of_plots)

        for i, element_graphic in enumerate(self.element_graphic_list):
            self.plot_element(self.axis_list[i], element_graphic)

        plt.show(block=block)
        if pause > 0:
            plt.pause(pause)

    def plot_element(self, axis, element_graphic):
        center_color = self.oil_color if element_graphic.phase_in_middle == OIL else self.water_color
        inner_layers_color = self.water_color
        outter_layers_color = self.oil_color

        axis.clear()
        self.plot_graphic(
            axis,
            element_graphic.name,
            element_graphic.vertices,
            element_graphic.inner_arcs,
            element_graphic.outter_arcs,
            center_color,
            inner_layers_color,
            outter_layers_color,
            self.line_color,
        )

    @staticmethod
    def reset_axes(number_of_plots):
        plt.close()
        fig, ax = plt.subplots(*ElementPlotter.find_2d_dimensions(number_of_plots))
        if number_of_plots > 1:
            axis_list = ax.flat
        else:
            axis_list = [ax]
        fig.suptitle("Elements")
        plt.grid(False)

        for axis in axis_list[number_of_plots:]:
            axis.axis("off")

        return axis_list[:number_of_plots]

    @staticmethod
    def plot_graphic(
        axis, name, vertices, inner_arcs, outter_arcs, center_color, inner_layers_color, outter_layers_color, line_color
    ):
        axis.fill(vertices[:, 0], vertices[:, 1], center_color)
        borders_x = np.concatenate((vertices[:, 0], [vertices[0][0]]))
        borders_y = np.concatenate((vertices[:, 1], [vertices[0][1]]))
        axis.plot(borders_x, borders_y, line_color, linewidth=1)  # Plot vertices in blue

        # draw oil layeres
        for arc_x, arc_y in outter_arcs:
            if arc_x is None:
                continue
            axis.fill(arc_x, arc_y, outter_layers_color)
            axis.plot(arc_x, arc_y, line_color, linewidth=1)

        # draw water layers
        for arc_x, arc_y in inner_arcs:
            if arc_x is None:
                continue
            axis.fill(arc_x, arc_y, inner_layers_color)
            axis.plot(arc_x, arc_y, line_color, linewidth=1)

        axis.set_title(name)
        axis.set_aspect("equal", adjustable="box")
        axis.label_outer()

    @staticmethod
    def find_2d_dimensions(n):
        """
        Find the 2D dimensions (rows, cols) for a given number of elements n such that the grid
        is as close to square as possible.

        Args:
        n (int): The number of elements.

        Returns:
        tuple: A tuple (rows, cols) representing the dimensions of the grid.
        """
        if n <= 0:
            return (0, 0)

        # Start with the assumption that the grid is as square as possible
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        return (rows, cols)


class ElementGraphic:
    def __init__(self, angles=[], radius=1, distance=[], theta=[], distance_o=[], theta_o=[], phase_in_middle=OIL):
        self.angles = np.radians(angles)
        self.radius = radius
        self.distance = distance
        self.theta = np.radians(theta)
        self.distance_o = distance_o
        self.theta_o = np.radians(theta_o)
        self.phase_in_middle = phase_in_middle

        self.vertices = None
        self.inner_arcs = None
        self.outter_arcs = None

        self.name = ""

    def set_name(self, name):
        self.name = name

    def update(self):
        self.vertices, self.inner_arcs, self.outter_arcs = self.draw_element(
            self.angles, self.radius, self.distance, self.theta, self.distance_o, self.theta_o
        )

    @staticmethod
    def draw_arc(p1, p2, angle1, angle2):
        """
        Draws a curved line between two points with specified exit and arrival angles.

        Parameters:
        - p1: tuple, (x, y) coordinates of the starting point.
        - p2: tuple, (x, y) coordinates of the ending point.
        - angle1: float, angle in radians of the tangent direction at the starting point.
        - angle2: float, angle in radians of the tangent direction at the ending point.
        """
        # Control points
        p1x, p1y = p1
        p2x, p2y = p2

        # Calculate the control points
        control_dist = np.linalg.norm(np.array(p2) - np.array(p1)) / 2  # Arbitrary choice for control points distance
        c1x = p1x + control_dist * np.cos(angle1)
        c1y = p1y + control_dist * np.sin(angle1)
        c2x = p2x + control_dist * np.cos(angle2 + np.pi)
        c2y = p2y + control_dist * np.sin(angle2 + np.pi)

        # Create the Bézier curve
        t = np.linspace(0, 1, 100)
        curve_x = (1 - t) ** 3 * p1x + 3 * (1 - t) ** 2 * t * c1x + 3 * (1 - t) * t**2 * c2x + t**3 * p2x
        curve_y = (1 - t) ** 3 * p1y + 3 * (1 - t) ** 2 * t * c1y + 3 * (1 - t) * t**2 * c2y + t**3 * p2y

        return curve_x, curve_y

    @staticmethod
    def draw_element(angles, radius, distance, theta, distance_o, theta_o):
        # Define vertices of the triangle
        if len(angles) == 0:
            vertices = ElementGraphic.circle_vertices(radius)
        elif len(angles) == 3:
            vertices = ElementGraphic.triangle_vertices(angles, radius)
        else:
            vertices = ElementGraphic.square_vertices(radius)

        inner_arcs = ElementGraphic.arcs_in_corners(vertices, distance, theta)
        outter_arcs = ElementGraphic.arcs_in_corners(vertices, distance_o, theta_o)

        return vertices, inner_arcs, outter_arcs

    @staticmethod
    def circle_vertices(radius):
        number_of_points = 20
        vertices = np.empty((number_of_points, 2))
        for i, angle in enumerate(np.linspace(0, 2 * np.pi, number_of_points)):
            vertices[i] = [np.sin(angle) * radius, np.cos(angle) * radius]
        return vertices

    @staticmethod
    def square_vertices(radius):
        A = (0, 0)
        B = (2 * radius, 0)
        C = (2 * radius, 2 * radius)
        D = (0, 2 * radius)
        vertices = np.array([A, B, C, D])

        return vertices

    @staticmethod
    def triangle_vertices(angles, radius):
        # Sum of angles must be π radians (180 degrees)
        if not np.isclose(np.sum(angles), np.pi):
            raise ValueError("The sum of the angles must be 180 degrees.")

        # Calculate the semi-perimeter (s) using the formula: s = r * (A + B + C) / 2
        s = radius * (np.sum(angles)) / 2

        # Calculate the sides of the triangle using the formula: side = 2 * r * tan(angle / 2)
        sides = 2 * radius * np.tan(angles / 2)

        # Calculate the coordinates of the vertices
        # Assume the first vertex is at the origin (0,0)
        A = (0, 0)

        # Second vertex at (side_a, 0)
        B = (sides[0], 0)

        # Third vertex calculated using law of cosines and coordinates geometry
        Cx = sides[1] * np.cos(angles[2])
        Cy = sides[1] * np.sin(angles[2])
        C = (Cx, Cy)

        # Define vertices of the triangle
        vertices = np.array([A, B, C])

        return vertices

    @staticmethod
    def point_on_side(v1, v2, dist):
        # Calculate points at a specified distance from a vertex along a side
        line_vec = np.array(v2) - np.array(v1)
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        return np.array(v1) + dist * line_unitvec

    @staticmethod
    def arcs_in_corners(vertices, distances, thetas):
        number_of_corners = len(vertices)
        if len(distances) != number_of_corners:
            return []

        opposite_thetas = np.pi - np.array(thetas)
        arcs = [(None, None)] * number_of_corners
        for i in range(number_of_corners):
            if distances[i] == 0:
                continue

            v1 = vertices[i]
            v2 = vertices[(i + 1) % number_of_corners]
            v3 = vertices[(i - 1) % number_of_corners]
            if (
                np.linalg.norm(np.array(v2) - np.array(v1)) >= distances[i]
                and np.linalg.norm(np.array(v3) - np.array(v1)) >= distances[i]
            ):
                p1 = ElementGraphic.point_on_side(v1, v2, distances[i])
                p2 = ElementGraphic.point_on_side(v1, v3, distances[i])

                dx = v2[0] - v1[0]
                dy = v2[1] - v1[1]
                angle2 = np.arctan2(dy, dx)
                dx = v3[0] - v1[0]
                dy = v3[1] - v1[1]
                angle3 = np.arctan2(dy, dx)

                arc_x, arc_y = ElementGraphic.draw_arc(
                    p1, p2, angle2 + opposite_thetas[i], np.pi + angle3 - opposite_thetas[i]
                )
                arc_x = np.append(arc_x, v1[0])
                arc_y = np.append(arc_y, v1[1])
                arcs[i] = (arc_x, arc_y)
        return arcs
