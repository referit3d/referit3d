'''
Created on December 8, 2016

@author: Panos Achlioptas and Lin Shao
@contact: pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''

import numpy as np
import warnings

l2_norm = np.linalg.norm


class Cuboid(object):
    '''
    A class representing a 3D Cuboid.
    '''

    def __init__(self, extrema):
        '''
        Constructor.
            Args: extrema (numpy array) containing 6 non-negative integers [xmin, ymin, zmin, xmax, ymax, zmax].
        '''
        self.extrema = extrema
        self.corners = self._corner_points()

    def __str__(self):
        return 'Cuboid with  [xmin, ymin, zmin, xmax, ymax, zmax] coordinates = %s.' % (str(self.extrema),)

    @property
    def extrema(self):
        return self._extrema

    @extrema.setter
    def extrema(self, value):
        self._extrema = value
        [xmin, ymin, zmin, xmax, ymax, zmax] = self._extrema
        if xmax == xmin or zmin == zmax or ymax == ymin:
            warnings.warn('Degenerate Cuboid was specified (its volume and/or area are zero).')
        if xmin > xmax or ymin > ymax or zmin > zmax:
            raise ValueError('Check extrema of cuboid.')

    def _corner_points(self):
        [xmin, ymin, zmin, xmax, ymax, zmax] = self.extrema
        c1 = np.array([xmin, ymin, zmin])
        c2 = np.array([xmax, ymin, zmin])
        c3 = np.array([xmax, ymax, zmin])
        c4 = np.array([xmin, ymax, zmin])
        c5 = np.array([xmin, ymin, zmax])
        c6 = np.array([xmax, ymin, zmax])
        c7 = np.array([xmax, ymax, zmax])
        c8 = np.array([xmin, ymax, zmax])
        return np.vstack([c1, c2, c3, c4, c5, c6, c7, c8])

    def diagonal_length(self):
        return l2_norm(self.extrema[:3] - self.extrema[3:])

    def get_extrema(self):
        ''' Syntactic sugar to get the extrema property into separate variables.
        '''
        e = self.extrema
        return e[0], e[1], e[2], e[3], e[4], e[5]

    def volume(self):
        [xmin, ymin, zmin, xmax, ymax, zmax] = self.extrema
        return (xmax - xmin) * (ymax - ymin) * (zmax - zmin)

    def height(self):
        [_, _, zmin, _, _, zmax] = self.extrema
        return zmax - zmin

    def intersection_with(self, other):
        [sxmin, symin, szmin, sxmax, symax, szmax] = self.get_extrema()
        [oxmin, oymin, ozmin, oxmax, oymax, ozmax] = other.get_extrema()
        dx = min(sxmax, oxmax) - max(sxmin, oxmin)
        dy = min(symax, oymax) - max(symin, oymin)
        dz = min(szmax, ozmax) - max(szmin, ozmin)
        inter = 0

        if (dx > 0) and (dy > 0) and (dz > 0):
            inter = dx * dy * dz

        return inter

    def barycenter(self):
        n_corners = self.corners.shape[0]
        return np.sum(self.corners, axis=0) / n_corners

    def faces(self):
        corners = self.corners
        [xmin, ymin, zmin, xmax, ymax, zmax] = self.extrema
        xmin_f = corners[corners[:, 0] == xmin, :]
        xmax_f = corners[corners[:, 0] == xmax, :]
        ymin_f = corners[corners[:, 1] == ymin, :]
        ymax_f = corners[corners[:, 1] == ymax, :]
        zmin_f = corners[corners[:, 2] == zmin, :]
        zmax_f = corners[corners[:, 2] == zmax, :]
        return [xmin_f, xmax_f, ymin_f, ymax_f, zmin_f, zmax_f]

    def z_bottom_face(self):
        return self.faces()[-2]

    def z_top_face(self):
        return self.faces()[-1]

    def is_point_inside(self, point):
        '''Given a 3D point tests if it lies inside the Cuboid.
        '''
        [xmin, ymin, zmin, xmax, ymax, zmax] = self.extrema
        return np.all([xmin, ymin, zmin] <= point) and np.all([xmax, ymax, zmax] >= point)

    def containing_sector(self, sector_center, ignore_z_axis=True):
        '''Computes the tightest (conic) sector that contains the Cuboid. The sector's center is defined by the user.
        Input:
            sector_center: 3D Point where the sector begins.
            ignore_z_axis: (Boolean) if True the Cuboid is treated as rectangle by eliminating it's z-dimension.
        Notes: Roughly it computes the angle between the ray's starting at the sector's center and each side of the cuboid.
        The one with the largest angle is the requested sector.
        '''
        if self.is_point_inside(sector_center):
            raise ValueError('Sector\'s center lies inside the bounding box.')

        def angle_of_sector(sector_center, side):
            x1, y1, x2, y2 = side
            line_1 = np.array([x1 - sector_center[0], y1 - sector_center[1]])  # First diagonal pair of points of cuboid
            line_2 = np.array([x2 - sector_center[0], y2 - sector_center[1]])
            cos = line_1.dot(line_2) / (l2_norm(line_1) * l2_norm(line_2))
            if cos >= 1 or cos <= -1:
                angle = 0
            else:
                angle = np.arccos(cos)
                assert (angle <= np.pi and angle >= 0)
            return angle

        if ignore_z_axis:
            [xmin, ymin, _, xmax, ymax, _] = self.extrema
            sides = [[xmin, ymin, xmax, ymax],
                     [xmax, ymin, xmin, ymax],
                     [xmin, ymax, xmax, ymax],
                     [xmin, ymin, xmax, ymin],
                     [xmin, ymin, xmin, ymax],
                     [xmax, ymin, xmax, ymax],
                     ]

            a0 = angle_of_sector(sector_center, sides[0])
            a1 = angle_of_sector(sector_center, sides[1])  # a0, a1: checking the diagonals.
            a2 = angle_of_sector(sector_center, sides[2])
            a3 = angle_of_sector(sector_center, sides[3])
            a4 = angle_of_sector(sector_center, sides[4])
            a5 = angle_of_sector(sector_center, sides[5])
            largest = np.argmax([a0, a1, a2, a3, a4, a5])
            return np.array(sides[largest][0:2]), np.array(sides[largest][2:])

    def union_with(self, other):
        return self.volume() + other.volume() - self.intersection_with(other)

    def iou_with(self, other):
        inter = self.intersection_with(other)
        union = self.union_with(other)
        return float(inter) / union

    def overlap_ratio_with(self, other, ratio_type='union'):
        '''
        Returns the overlap ratio between two cuboids. That is the ratio of their volume intersection
        and their overlap. If the ratio_type is 'union' then the overlap is the volume of their union. If it is min, it
        the min volume between them.
        '''
        inter = self.intersection_with(other)
        if ratio_type == 'union':
            union = self.union_with(other)
            return float(inter) / union
        elif ratio_type == 'min':
            return float(inter) / min(self.volume(), other.volume())
        else:
            ValueError('ratio_type must be either \'union\', or \'min\'.')

    def plot(self, axis=None, c='r'):
        """Plot the Cuboid.
        Input:
            axis - (matplotlib.axes.Axes) where the cuboid will be drawn.
            c - (String) specifying the color of the cuboid. Must be valid for matplotlib.pylab.plot
        """
        corners = self.corners
        if axis is not None:
            axis.plot([corners[0, 0], corners[1, 0]], [corners[0, 1], corners[1, 1]], zs=[corners[0, 2], corners[1, 2]],
                      c=c)
            axis.plot([corners[1, 0], corners[2, 0]], [corners[1, 1], corners[2, 1]], zs=[corners[1, 2], corners[2, 2]],
                      c=c)
            axis.plot([corners[2, 0], corners[3, 0]], [corners[2, 1], corners[3, 1]], zs=[corners[2, 2], corners[3, 2]],
                      c=c)
            axis.plot([corners[3, 0], corners[0, 0]], [corners[3, 1], corners[0, 1]], zs=[corners[3, 2], corners[0, 2]],
                      c=c)
            axis.plot([corners[4, 0], corners[5, 0]], [corners[4, 1], corners[5, 1]], zs=[corners[4, 2], corners[5, 2]],
                      c=c)
            axis.plot([corners[5, 0], corners[6, 0]], [corners[5, 1], corners[6, 1]], zs=[corners[5, 2], corners[6, 2]],
                      c=c)
            axis.plot([corners[6, 0], corners[7, 0]], [corners[6, 1], corners[7, 1]], zs=[corners[6, 2], corners[7, 2]],
                      c=c)
            axis.plot([corners[7, 0], corners[4, 0]], [corners[7, 1], corners[0, 1]], zs=[corners[7, 2], corners[4, 2]],
                      c=c)
            axis.plot([corners[0, 0], corners[4, 0]], [corners[0, 1], corners[4, 1]], zs=[corners[0, 2], corners[4, 2]],
                      c=c)
            axis.plot([corners[1, 0], corners[5, 0]], [corners[1, 1], corners[5, 1]], zs=[corners[1, 2], corners[5, 2]],
                      c=c)
            axis.plot([corners[2, 0], corners[6, 0]], [corners[2, 1], corners[6, 1]], zs=[corners[2, 2], corners[6, 2]],
                      c=c)
            axis.plot([corners[3, 0], corners[7, 0]], [corners[3, 1], corners[7, 1]], zs=[corners[3, 2], corners[7, 2]],
                      c=c)
            return axis.figure
        else:
            ValueError('NYI')

    @staticmethod
    def from_corner_points_to_cuboid(corners):
        xmax = np.max(corners[:, 0])
        xmin = np.min(corners[:, 0])
        ymax = np.max(corners[:, 1])
        ymin = np.min(corners[:, 1])
        zmax = np.max(corners[:, 2])
        zmin = np.min(corners[:, 2])
        extrema = [xmin, ymin, zmin, xmax, ymax, zmax]
        return Cuboid(extrema)

    @staticmethod
    def bounding_box_of_3d_points(points):
        xmin = np.min(points[:, 0])
        xmax = np.max(points[:, 0])
        ymin = np.min(points[:, 1])
        ymax = np.max(points[:, 1])
        zmin = np.min(points[:, 2])
        zmax = np.max(points[:, 2])
        return Cuboid(np.array([xmin, ymin, zmin, xmax, ymax, zmax]))


class OrientedCuboid(object):
    def __init__(self, cx, cy, cz, lx, ly, lz, rot):
        """
        Constructor
        :param cx: center point x coordinate
        :param cy: center point y coordinate
        :param cz: center point z coordinate
        :param lx: length in the x direction
        :param ly: length in the y direction
        :param lz: length in the z direction
        :param rot: Rotation around z axis matrix [4x4]
        """
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.rot = np.array(rot).reshape(3, 3)
        self.corners = self._corners()
        self.extrema = self._extrema()

    def _extrema(self):
        xmin = self.cx - self.lx / 2.0
        xmax = self.cx + self.lx / 2.0
        ymin = self.cy - self.ly / 2.0
        ymax = self.cy + self.ly / 2.0
        zmin = self.cz - self.lz / 2.0
        zmax = self.cz + self.lz / 2.0
        return np.array([xmin, ymin, zmin, xmax, ymax, zmax])

    def inverse_rotation_matrix(self, translate=True):
        rotation = np.eye(4)
        rotation[:3, :3] = self.rot.transpose()

        if translate:
            rotation[:3, 3] = [self.cx, self.cy, self.cz]

        return rotation

    def _corners(self):
        # get the axis aligned corners
        axis_aligned_corners = self.axis_aligned_corners()

        # calculate the relative coordinates to the center of axis_aligned_bbox
        axis_aligned_corners = axis_aligned_corners - [self.cx, self.cy, self.cz]

        # transform the points (apply z rotation) also plus the center coordinates.
        axis_aligned_corners = np.hstack([axis_aligned_corners, np.ones((axis_aligned_corners.shape[0], 1))])
        rotation = np.eye(4)
        rotation[:3, :3] = self.rot.copy()
        rotation[:3, 3] = [self.cx, self.cy, self.cz]

        corners = np.dot(rotation, axis_aligned_corners.T).T[:, 0:3]

        return corners

    def axis_aligned_corners(self):
        [xmin, ymin, zmin, xmax, ymax, zmax] = self._extrema()
        c1 = np.array([xmin, ymin, zmin])
        c2 = np.array([xmax, ymin, zmin])
        c3 = np.array([xmax, ymax, zmin])
        c4 = np.array([xmin, ymax, zmin])
        c5 = np.array([xmin, ymin, zmax])
        c6 = np.array([xmax, ymin, zmax])
        c7 = np.array([xmax, ymax, zmax])
        c8 = np.array([xmin, ymax, zmax])
        axis_aligned_corners = np.vstack([c1, c2, c3, c4, c5, c6, c7, c8])
        return axis_aligned_corners

    def center(self):
        return np.array([self.cx, self.cy, self.cz])

    def size(self):
        return np.array([self.lx, self.ly, self.lz])

    def z_faces(self):
        corners = self._corners()
        [_, _, zmin, _, _, zmax] = self._extrema()
        bottom_face = corners[np.array(corners[:, 2], np.float32) == np.array(zmin, np.float32), :]
        top_face = corners[np.array(corners[:, 2], np.float32) == np.array(zmax, np.float32), :]
        return [bottom_face, top_face]

    def plot(self, axis=None, c='r'):
        """ Plot the Cuboid.
        Input:
            axis - (matplotlib.axes.Axes) where the cuboid will be drawn.
            c - (String) specifying the color of the cuboid. Must be valid for matplotlib.pylab.plot
        """
        cors = self.corners
        if axis is not None:
            axis.plot([cors[0, 0], cors[1, 0]], [cors[0, 1], cors[1, 1]], zs=[cors[0, 2], cors[1, 2]], c=c)
            axis.plot([cors[1, 0], cors[2, 0]], [cors[1, 1], cors[2, 1]], zs=[cors[1, 2], cors[2, 2]], c=c)
            axis.plot([cors[2, 0], cors[3, 0]], [cors[2, 1], cors[3, 1]], zs=[cors[2, 2], cors[3, 2]], c=c)
            axis.plot([cors[3, 0], cors[0, 0]], [cors[3, 1], cors[0, 1]], zs=[cors[3, 2], cors[0, 2]], c=c)
            axis.plot([cors[4, 0], cors[5, 0]], [cors[4, 1], cors[5, 1]], zs=[cors[4, 2], cors[5, 2]], c=c)
            axis.plot([cors[5, 0], cors[6, 0]], [cors[5, 1], cors[6, 1]], zs=[cors[5, 2], cors[6, 2]], c=c)
            axis.plot([cors[6, 0], cors[7, 0]], [cors[6, 1], cors[7, 1]], zs=[cors[6, 2], cors[7, 2]], c=c)
            axis.plot([cors[7, 0], cors[4, 0]], [cors[7, 1], cors[0, 1]], zs=[cors[7, 2], cors[4, 2]], c=c)
            axis.plot([cors[0, 0], cors[4, 0]], [cors[0, 1], cors[4, 1]], zs=[cors[0, 2], cors[4, 2]], c=c)
            axis.plot([cors[1, 0], cors[5, 0]], [cors[1, 1], cors[5, 1]], zs=[cors[1, 2], cors[5, 2]], c=c)
            axis.plot([cors[2, 0], cors[6, 0]], [cors[2, 1], cors[6, 1]], zs=[cors[2, 2], cors[6, 2]], c=c)
            axis.plot([cors[3, 0], cors[7, 0]], [cors[3, 1], cors[7, 1]], zs=[cors[3, 2], cors[7, 2]], c=c)
            return axis.figure
        else:
            ValueError('NYI')

    def intersection_with(self, other):
        if np.any(self.rot != np.eye(3)):
            raise NotImplementedError("intersection_with(): Not implemeted for oriented boxes")
        [sxmin, symin, szmin, sxmax, symax, szmax] = self.extrema
        [oxmin, oymin, ozmin, oxmax, oymax, ozmax] = other.extrema
        dx = min(sxmax, oxmax) - max(sxmin, oxmin)
        dy = min(symax, oymax) - max(symin, oymin)
        dz = min(szmax, ozmax) - max(szmin, ozmin)
        inter = 0

        if (dx > 0) and (dy > 0) and (dz > 0):
            inter = dx * dy * dz

        return inter

    def volume(self):
        [xmin, ymin, zmin, xmax, ymax, zmax] = self.extrema
        return (xmax - xmin) * (ymax - ymin) * (zmax - zmin)


def iou_3d(a, b):
    """Computes IoU of two axis aligned bboxes.
    Args:
        box_a, box_b: 6D of center and lengths
    Returns:
        iou
    """
    # a = Cuboid.from_corner_points_to_cuboid(a)
    # b = Cuboid.from_corner_points_to_cuboid(b)
    # iou = a.iou_with(b)
    # intersection = a.intersection_with(b)
    # vol_a = a.volume()
    # vol_b = b.volume()

    max_a = np.max(a, axis=0)
    max_b = np.max(b, axis=0)
    min_max = np.array([max_a, max_b]).min(0)

    min_a = np.min(a, axis=0)
    min_b = np.min(b, axis=0)
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()

    vol_a = (max_a - min_a).prod()
    vol_b = (max_b - min_b).prod()
    union = vol_a + vol_b - intersection

    iou = 1.0 * intersection / union

    return iou, intersection, vol_a, vol_b
