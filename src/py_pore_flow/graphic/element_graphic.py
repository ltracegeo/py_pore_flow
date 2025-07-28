import numpy as np


WATER = 0
OIL = 1


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
        self.description = ""

    def set_name(self, name):
        self.name = name

    def set_description(self, description):
        self.description = description

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
        outter_arcs = ElementGraphic.arcs_in_corners(vertices, distance_o, np.pi - theta_o)

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

        half_angles = angles / 2
        half_lengths = radius / np.tan(half_angles)
        A = (0, 0)
        B = (half_lengths[0] + half_lengths[1], 0)
        secound_length = half_lengths[0] + half_lengths[2]
        C = (secound_length * np.cos(angles[0]), secound_length * np.sin(angles[0]))

        return np.array([A, B, C])

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
