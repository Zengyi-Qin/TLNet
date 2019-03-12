#!/usr/bin/env python3.5
import numpy as np
from PIL import Image, ImageDraw

import os

from avod.core import calib_utils


class ObjectLabel:
    """Object Label Class
    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                      'Misc' or 'DontCare'

    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                      truncated refers to the object leaving image boundaries

    1    occluded     Integer (0,1,2,3) indicating occlusion state:
                      0 = fully visible, 1 = partly occluded
                      2 = largely occluded, 3 = unknown

    1    alpha        Observation angle of object, ranging [-pi..pi]

    4    bbox         2D bounding box of object in the image (0-based index):
                      contains left, top, right, bottom pixel coordinates

    3    dimensions   3D object dimensions: height, width, length (in meters)

    3    location     3D object location x,y,z in camera coordinates (in meters)

    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]

    1    score        Only for results: Float, indicating confidence in
                      detection, needed for p/r curves, higher is better.
    """

    def __init__(self):
        self.type = ""  # Type of object
        self.truncation = 0.
        self.occlusion = 0.
        self.alpha = 0.
        self.x1 = 0.
        self.y1 = 0.
        self.x2 = 0.
        self.y2 = 0.
        self.h = 0.
        self.w = 0.
        self.l = 0.
        self.t = (0., 0., 0.)
        self.ry = 0.
        self.score = 0.

    def __eq__(self, other):
        """Compares the given object to the current ObjectLabel instance.

        :param other: object to compare to this instance against
        :return: True, if other and current instance is the same
        """
        if not isinstance(other, ObjectLabel):
            return False

        if self.__dict__ != other.__dict__:
            return False
        else:
            return True


def read_labels(label_dir, img_idx, results=False):
    """Reads in label data file from Kitti Dataset.

    Returns:
    obj_list -- List of instances of class ObjectLabel.

    Keyword arguments:
    label_dir -- directory of the label files
    img_idx -- index of the image
    """

    # Define the object list
    obj_list = []

    # Extract the list
    if os.stat(label_dir + "/%06d.txt" % img_idx).st_size == 0:
        return

    if results:
        p = np.loadtxt(label_dir + "/%06d.txt" % img_idx, delimiter=' ',
                       dtype=str,
                       usecols=np.arange(start=0, step=1, stop=16))
    else:
        p = np.loadtxt(label_dir + "/%06d.txt" % img_idx, delimiter=' ',
                       dtype=str,
                       usecols=np.arange(start=0, step=1, stop=15))

    # Check if the output is single dimensional or multi dimensional
    if len(p.shape) > 1:
        label_num = p.shape[0]
    else:
        label_num = 1

    for idx in np.arange(label_num):
        obj = ObjectLabel()

        if label_num > 1:
            # Fill in the object list
            obj.type = p[idx, 0]
            obj.truncation = float(p[idx, 1])
            obj.occlusion = float(p[idx, 2])
            obj.alpha = float(p[idx, 3])
            obj.x1 = float(p[idx, 4])
            obj.y1 = float(p[idx, 5])
            obj.x2 = float(p[idx, 6])
            obj.y2 = float(p[idx, 7])
            obj.h = float(p[idx, 8])
            obj.w = float(p[idx, 9])
            obj.l = float(p[idx, 10])
            obj.t = (float(p[idx, 11]), float(p[idx, 12]), float(p[idx, 13]))
            obj.ry = float(p[idx, 14])
            if results:
                obj.score = float(p[idx, 15])
            else:
                obj.score = 0.0
        else:
            # Fill in the object list
            obj.type = p[0]
            obj.truncation = float(p[1])
            obj.occlusion = float(p[2])
            obj.alpha = float(p[3])
            obj.x1 = float(p[4])
            obj.y1 = float(p[5])
            obj.x2 = float(p[6])
            obj.y2 = float(p[7])
            obj.h = float(p[8])
            obj.w = float(p[9])
            obj.l = float(p[10])
            obj.t = (float(p[11]), float(p[12]), float(p[13]))
            obj.ry = float(p[14])
            if results:
                obj.score = float(p[15])
            else:
                obj.score = 0.0

        obj_list.append(obj)

    return obj_list


def build_bbs_from_objects(obj_list, class_needed):
    """ Converts between a list of objects and a numpy array containing the
        bounding boxes.

     :param obj_list: an object list as per object class
     :param class_needed: 'Car', 'Pedestrian' ...  If no class filtering is
        needed use 'All'

     :return boxes_2d : a numpy array formed as a list of boxes in the form
        [boxes_frame_1, ... boxes_frame_n], where boxes_frame_n is a numpy
        array containing all bounding boxes in the frame n with the format:
        [[x1, y1, x2, y2], [x1, y1, x2, y2]].

    :return boxes_3d : a numpy array formed as a list of boxes in the form
        [boxes_frame_1, ... boxes_frame_n], where boxes_frame_n is a numpy
        array containing all bounding boxes in the frame n with the format:
        [[ry, l, h, w, tx, ty, tz],...[ry, l, h, w, tx, ty, tz]]

    :return scores : a numpy array of the form
        [[scores_frame_1],
         ...,
         [scores_frame_n]]
     """

    if class_needed == 'All':
        obj_detections = obj_list
    else:
        if isinstance(class_needed, str):
            obj_detections = [detections for detections in obj_list if
                              detections.type == class_needed]
        elif isinstance(class_needed, list):
            obj_detections = [detections for detections in obj_list if
                              detections.type in class_needed]
        else:
            raise TypeError("Invalid type for class_needed, {} should be "
                            "str or list".format(type(class_needed)))

    # Build A Numpy Array Of 2D Bounding Boxes
    x1 = [obj.x1 for obj in obj_detections]
    y1 = [obj.y1 for obj in obj_detections]
    x2 = [obj.x2 for obj in obj_detections]
    y2 = [obj.y2 for obj in obj_detections]

    ry = [obj.ry for obj in obj_detections]
    l = [obj.l for obj in obj_detections]
    h = [obj.h for obj in obj_detections]
    w = [obj.w for obj in obj_detections]
    tx = [obj.t[0] for obj in obj_detections]
    ty = [obj.t[1] for obj in obj_detections]
    tz = [obj.t[2] for obj in obj_detections]
    scores = [obj.score for obj in obj_detections]

    num_objs = len(obj_detections)
    boxes_2d = np.zeros((num_objs, 4))
    boxes_3d = np.zeros((num_objs, 7))  # [ry, l, h, w, tx, ty, tz]

    for it in range(num_objs):
        boxes_2d[it] = np.array([x1[it],
                                 y1[it],
                                 x2[it],
                                 y2[it]])

        boxes_3d[it] = np.array([ry[it],
                                 l[it],
                                 h[it],
                                 w[it],
                                 tx[it],
                                 ty[it],
                                 tz[it]])

    return boxes_2d, boxes_3d, scores


def get_lidar_point_cloud(img_idx, calib_dir, velo_dir,
                          im_size=None, min_intensity=None):
    """ Calculates the lidar point cloud, and optionally returns only the
    points that are projected to the image.

    :param img_idx: image index
    :param calib_dir: directory with calibration files
    :param velo_dir: directory with velodyne files
    :param im_size: (optional) 2 x 1 list containing the size of the image
                      to filter the point cloud [w, h]
    :param min_intensity: (optional) minimum intensity required to keep a point

    :return: (3, N) point_cloud in the form [[x,...][y,...][z,...]]
    """

    # Read calibration info
    frame_calib = calib_utils.read_calibration(calib_dir, img_idx)
    x, y, z, i = calib_utils.read_lidar(velo_dir=velo_dir, img_idx=img_idx)

    # Calculate the point cloud
    pts = np.vstack((x, y, z)).T
    pts = calib_utils.lidar_to_cam_frame(pts, frame_calib)

    # The given image is assumed to be a 2D image
    if not im_size:
        point_cloud = pts.T
        return point_cloud

    else:
        # Only keep points in front of camera (positive z)
        pts = pts[pts[:, 2] > 0]
        point_cloud = pts.T

        # Project to image frame
        point_in_im = calib_utils.project_to_image(point_cloud, p=frame_calib.p2).T

        # Filter based on the given image size
        image_filter = (point_in_im[:, 0] > 0) & \
                       (point_in_im[:, 0] < im_size[0]) & \
                       (point_in_im[:, 1] > 0) & \
                       (point_in_im[:, 1] < im_size[1])

    if not min_intensity:
        return pts[image_filter].T

    else:
        intensity_filter = i > min_intensity
        point_filter = np.logical_and(image_filter, intensity_filter)
        return pts[point_filter].T


def get_road_plane(img_idx, planes_dir):
    """Reads the road plane from file

    :param int img_idx : Index of image
    :param str planes_dir : directory containing plane text files

    :return plane : List containing plane equation coefficients
    """

    plane_file = planes_dir + '/%06d.txt' % img_idx

    with open(plane_file, 'r') as input_file:
        lines = input_file.readlines()
        input_file.close()

    # Plane coefficients stored in 4th row
    lines = lines[3].split()

    # Convert str to float
    lines = [float(i) for i in lines]

    plane = np.asarray(lines)

    # Ensure normal is always facing up.
    # In Kitti's frame of reference, +y is down
    if plane[1] > 0:
        plane = -plane

    # Normalize the plane coefficients
    norm = np.linalg.norm(plane[0:3])
    plane = plane / norm

    return plane


def compute_box_corners_3d(object_label):
    """Computes the 3D bounding box corner positions from an ObjectLabel

    :param object_label: ObjectLabel to compute corners from
    :return: a numpy array of 3D corners if the box is in front of the camera,
             an empty array otherwise
    """

    # Compute rotational matrix
    rot = np.array([[+np.cos(object_label.ry), 0, +np.sin(object_label.ry)],
                    [0, 1, 0],
                    [-np.sin(object_label.ry), 0, +np.cos(object_label.ry)]])

    l = object_label.l
    w = object_label.w
    h = object_label.h

    # 3D BB corners
    x_corners = np.array(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

    corners_3d = np.dot(rot, np.array([x_corners, y_corners, z_corners]))

    corners_3d[0, :] = corners_3d[0, :] + object_label.t[0]
    corners_3d[1, :] = corners_3d[1, :] + object_label.t[1]
    corners_3d[2, :] = corners_3d[2, :] + object_label.t[2]

    return corners_3d


def project_box3d_to_image(corners_3d, p):
    """Computes the 3D bounding box projected onto
    image space.

    Keyword arguments:
    obj -- object file to draw bounding box
    p -- transform matrix

    Returns:
        corners : numpy array of corner points projected
        onto image space.
        face_idx: numpy array of 3D bounding box face
    """
    # index for 3d bounding box face
    # it is converted to 4x4 matrix
    face_idx = np.array([0, 1, 5, 4,  # front face
                         1, 2, 6, 5,  # left face
                         2, 3, 7, 6,  # back face
                         3, 0, 4, 7]).reshape((4, 4))  # right face
    return calib_utils.project_to_image(corners_3d, p), face_idx


def compute_orientation_3d(obj, p):
    """Computes the orientation given object and camera matrix

    Keyword arguments:
    obj -- object file to draw bounding box
    p -- transform matrix
    """

    # compute rotational matrix
    rot = np.array([[+np.cos(obj.ry), 0, +np.sin(obj.ry)],
                    [0, 1, 0],
                    [-np.sin(obj.ry), 0, +np.cos(obj.ry)]])

    orientation3d = np.array([0.0, obj.l, 0.0, 0.0, 0.0, 0.0]).reshape(3, 2)
    orientation3d = np.dot(rot, orientation3d)

    orientation3d[0, :] = orientation3d[0, :] + obj.t[0]
    orientation3d[1, :] = orientation3d[1, :] + obj.t[1]
    orientation3d[2, :] = orientation3d[2, :] + obj.t[2]

    # only draw for boxes that are in front of the camera
    for idx in np.arange(orientation3d.shape[1]):
        if orientation3d[2, idx] < 0.1:
            return None

    return calib_utils.project_to_image(orientation3d, p)


def is_point_inside(points, box_corners):
    """Check if each point in a 3D point cloud lies within the 3D bounding box

    If we think of the bounding box as having bottom face
    defined by [P1, P2, P3, P4] and top face [P5, P6, P7, P8]
    then there are three directions on a perpendicular edge:
        u = P1 - P2
        v = P1 - P4
        w = P1 - P5

    A point x lies within the box when the following constraints
    are respected:
        - The dot product u.x is between u.P1 and u.P2
        - The dot product v.x is between v.P1 and v.P4
        - The dot product w.x is between w.P1 and w.P5

    :param points: (3, N) point cloud to test in the form
        [[x1...xn], [y1...yn], [z1...zn]]
    :param box_corners: 3D corners of the bounding box

    :return bool mask of which points are within the bounding box.
            Use numpy function .all() to check all points
    """

    p1 = box_corners[:, 0]
    p2 = box_corners[:, 1]
    p4 = box_corners[:, 3]
    p5 = box_corners[:, 4]

    u = p2 - p1
    v = p4 - p1
    w = p5 - p1

    # if u.P1 < u.x < u.P2
    u_dot_x = np.dot(u, points)
    u_dot_p1 = np.dot(u, p1)
    u_dot_p2 = np.dot(u, p2)

    # if v.P1 < v.x < v.P4
    v_dot_x = np.dot(v, points)
    v_dot_p1 = np.dot(v, p1)
    v_dot_p2 = np.dot(v, p4)

    # if w.P1 < w.x < w.P5
    w_dot_x = np.dot(w, points)
    w_dot_p1 = np.dot(w, p1)
    w_dot_p2 = np.dot(w, p5)

    point_mask = (u_dot_p1 < u_dot_x) & (u_dot_x < u_dot_p2) & \
                 (v_dot_p1 < v_dot_x) & (v_dot_x < v_dot_p2) & \
                 (w_dot_p1 < w_dot_x) & (w_dot_x < w_dot_p2)

    return point_mask


def get_point_filter(point_cloud, extents, ground_plane=None, offset_dist=2.0):
    """
    Creates a point filter using the 3D extents and ground plane

    :param point_cloud: Point cloud in the form [[x,...],[y,...],[z,...]]
    :param extents: 3D area in the form
        [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
    :param ground_plane: Optional, coefficients of the ground plane
        (a, b, c, d)
    :param offset_dist: If ground_plane is provided, removes points above
        this offset from the ground_plane
    :return: A binary mask for points within the extents and offset plane
    """

    point_cloud = np.asarray(point_cloud)

    # Filter points within certain xyz range
    x_extents = extents[0]
    y_extents = extents[1]
    z_extents = extents[2]

    extents_filter = (point_cloud[0] > x_extents[0]) & \
                     (point_cloud[0] < x_extents[1]) & \
                     (point_cloud[1] > y_extents[0]) & \
                     (point_cloud[1] < y_extents[1]) & \
                     (point_cloud[2] > z_extents[0]) & \
                     (point_cloud[2] < z_extents[1])

    if ground_plane is not None:
        ground_plane = np.array(ground_plane)

        # Calculate filter using ground plane
        ones_col = np.ones(point_cloud.shape[1])
        padded_points = np.vstack([point_cloud, ones_col])

        offset_plane = ground_plane + [0, 0, 0, -offset_dist]

        # Create plane filter
        dot_prod = np.dot(offset_plane, padded_points)
        plane_filter = dot_prod < 0

        # Combine the two filters
        point_filter = np.logical_and(extents_filter, plane_filter)
    else:
        # Only use the extents for filtering
        point_filter = extents_filter

    return point_filter

def two_d_iou(box, boxes):
    """Compute 2D IOU between a 2D bounding box 'box' and a list

    :param box: a numpy array in the form of [x1, y1, x2, y2] where (x1,y1) are
    image coordinates of the top-left corner of the bounding box, and (x2,y2)
    are the image coordinates of the bottom-right corner of the bounding box.

    :param boxes: a numpy array formed as a list of boxes in the form
    [[x1, y1, x2, y2], [x1, y1, x2, y2]].

    :return iou: a numpy array containing 2D IOUs between box and every element
    in numpy array boxes.
    """
    iou = np.zeros(len(boxes), np.float64)

    x1_int = np.maximum(box[0], boxes[:, 0])
    y1_int = np.maximum(box[1], boxes[:, 1])
    x2_int = np.minimum(box[2], boxes[:, 2])
    y2_int = np.minimum(box[3], boxes[:, 3])

    w_int = x2_int - x1_int
    h_int = y2_int - y1_int

    non_empty = np.logical_and(w_int > 0, h_int > 0)

    if non_empty.any():
        intersection_area = np.multiply(w_int[non_empty], h_int[non_empty])

        box_area = (box[2] - box[0]) * (box[3] - box[1])

        boxes_area = np.multiply(
            boxes[non_empty, 2] - boxes[non_empty, 0],
            boxes[non_empty, 3] - boxes[non_empty, 1])

        union_area = box_area + boxes_area - intersection_area

        iou[non_empty] = intersection_area / union_area

    return iou.round(3)


def three_d_iou(box, boxes):
    """Computes approximate 3D IOU between a 3D bounding box 'box' and a list
    of 3D bounding boxes 'boxes'. All boxes are assumed to be aligned with
    respect to gravity. Boxes are allowed to rotate only around their z-axis.

    :param box: a numpy array of the form: [ry, l, h, w, tx, ty, tz]
    :param boxes: a numpy array of the form:
        [[ry, l, h, w, tx, ty, tz], [ry, l, h, w, tx, ty, tz]]

    :return iou: a numpy array containing 3D IOUs between box and every element
        in numpy array boxes.
    """
    # First, rule out boxes that do not intersect by checking if the spheres
    # which inscribes them intersect.

    if len(boxes.shape) == 1:
        boxes = np.array([boxes])

    box_diag = np.sqrt(np.square(box[1]) +
                       np.square(box[2]) +
                       np.square(box[3])) / 2

    boxes_diag = np.sqrt(np.square(boxes[:, 1]) +
                         np.square(boxes[:, 2]) +
                         np.square(boxes[:, 3])) / 2

    dist = np.sqrt(np.square(boxes[:, 4] - box[4]) +
                   np.square(boxes[:, 5] - box[5]) +
                   np.square(boxes[:, 6] - box[6]))

    non_empty = box_diag + boxes_diag >= dist

    iou = np.zeros(len(boxes), np.float64)

    if non_empty.any():
        height_int, _ = height_metrics(box, boxes[non_empty])
        rect_int = get_rectangular_metrics(box, boxes[non_empty])

        intersection = np.multiply(height_int, rect_int)

        vol_box = np.prod(box[1:4])

        vol_boxes = np.prod(boxes[non_empty, 1:4], axis=1)

        union = vol_box + vol_boxes - intersection

        iou[non_empty] = intersection / union

    if iou.shape[0] == 1:
        iou = iou[0]

    return iou


def height_metrics(box, boxes):
    """Compute 3D height intersection and union between a box and a list of
    boxes

    :param box: a numpy array of the form: [ry, l, h, w, tx, ty, tz]

    :param boxes: a numpy array of the form: [[ry, l, h, w, tx, ty, tz],.....
                                        [ry, l, h, w, tx, ty, tz]]

    :return height_intersection: a numpy array containing the intersection along
    the gravity axis between the two bbs

    :return height_union: a numpy array containing the union along the gravity
    axis between the two bbs
    """
    boxes_heights = boxes[:, 2]
    boxes_centroid_heights = boxes[:, 5]

    min_y_boxes = boxes_centroid_heights - boxes_heights

    max_y_box = box[5]
    min_y_box = box[5] - box[2]

    max_of_mins = np.maximum(min_y_box, min_y_boxes)
    min_of_maxs = np.minimum(max_y_box, boxes_centroid_heights)

    offsets = min_of_maxs - max_of_mins
    height_intersection = np.maximum(0, offsets)

    height_union = np.maximum(min_y_box, boxes_centroid_heights) \
        - np.minimum(min_y_box, min_y_boxes) - \
        np.maximum(0, -offsets)

    return height_intersection, height_union


def get_rotated_3d_bb(boxes):
    """Compute rotated 3D bounding box coordinates.

    :param boxes: a numpy array of the form: [[ry, l, h, w, tx, ty, tz],...
                                         [ry, l, h, w, tx, ty, tz]]

    :return x: x coordinates of the four corners required to describe a 3D
    bounding box arranged as [[x1, x2, x3, x4],
                     [x1, x2, x3, x4],
                     ... ]

    :return z: z coordinates of the four corners required to describe a 3D
    bounding box arranged as [[z1, z2, z3, z4],
                     [z1, z2, z3, z4],
                     ... ].
    """

    if len(boxes.shape) == 1:
        boxes = np.array([boxes])

    x = np.array([[]])
    z = np.array([[]])

    for i in boxes:
        rot_mat = np.array([[np.cos(i[0]), np.sin(i[0])],
                            [-np.sin(i[0]), np.cos(i[0])]])

        x_corners = np.multiply(i[1] / 2, np.array([1, 1, -1, -1]))
        z_corners = np.multiply(i[3] / 2, np.array([1, -1, -1, 1]))

        temp_coor = np.dot(rot_mat, np.array([x_corners, z_corners]))

        # At the very first iteration, initialize x
        if x.shape[1] < 1:
            x = temp_coor[:1] + i[4]
            z = temp_coor[1:2] + i[6]
        # After that, append to the existing x
        else:
            x = np.append(x, temp_coor[:1] + i[4], axis=0)
            z = np.append(z, temp_coor[1:2] + i[6], axis=0)

    if x.shape[0] == 1:
        x = x[0]
        z = z[0]

    return x, z


def get_rectangular_metrics(box, boxes):
    """ Computes the intersection of the bases of oriented 3D bounding "box"
    and a set boxes of oriented 3D bounding boxes "boxes".

    :param box: a numpy array of the form: [ry, l, h, w, tx, ty, tz]

    :param boxes: a numpy array of the form: [[ry, l, h, w, tx, ty, tz],.....
                                        [ry, l, h, w, tx, ty, tz]]

    :return intersection: a numpy array containing intersection between the
    base of box and all other boxes.
    """
    if len(boxes.shape) == 1:
        boxes = np.array([boxes])

    mask_res = 0.01

    x_box, z_box = get_rotated_3d_bb(box)
    max_x_box = np.max(x_box)
    min_x_box = np.min(x_box)
    max_z_box = np.max(z_box)
    min_z_box = np.min(z_box)

    x_boxes, z_boxes = get_rotated_3d_bb(boxes)

    intersection = np.zeros(np.size(boxes, 0))

    if np.size(np.shape(x_boxes)) == 1:
        x_boxes = np.array([x_boxes])
        z_boxes = np.array([z_boxes])

    for i in range(np.size(boxes, 0)):
        x_i = x_boxes[i, :]
        z_i = z_boxes[i, :]
        test = max_x_box < np.min(x_i) or np.max(x_i) < min_x_box \
            or max_z_box < np.min(z_i) or np.max(z_i) < min_z_box

        if test:
            continue

        x_all = np.append(x_box, x_i)
        z_all = np.append(z_box, z_i)
        maxs = np.array([np.max(x_all), np.max(z_all)])
        mins = np.array([np.min(x_all), np.min(z_all)])

        mask_dims = np.int32(np.ceil((maxs - mins) / mask_res))

        mask_box_x = (x_box - mins[0]) / mask_res
        mask_box_z = (z_box - mins[1]) / mask_res
        mask_i_x = (x_i - mins[0]) / mask_res
        mask_i_z = (z_i - mins[1]) / mask_res
        # Drawing a binary image of the base of the two bounding boxes.
        # Then compute the element wise and of the two images to get the intersection.
        # Minor precision loss due to discretization.
        img = Image.new('L', (mask_dims[0], mask_dims[1]), 0)
        draw = ImageDraw.Draw(img, 'L')
        rect_coordinates = np.reshape(np.transpose(np.array([mask_box_x,
                                                             mask_box_z])), 8)
        rect_coordinates = np.append(rect_coordinates, rect_coordinates[0:2])
        draw.polygon(rect_coordinates.ravel().tolist(), outline=255, fill=255)
        del draw
        mask_box = np.asarray(img)

        img2 = Image.new('L', (mask_dims[0], mask_dims[1]), 0)
        draw = ImageDraw.Draw(img2, 'L')
        i_coordinates = np.reshape(np.transpose(np.array([mask_i_x,
                                                          mask_i_z])), 8)
        i_coordinates = np.append(i_coordinates, i_coordinates[0:2])
        draw.polygon(i_coordinates.ravel().tolist(), outline=255, fill=255)
        del draw
        mask_i = np.asarray(img2)

        mask_intersection = np.logical_and(mask_box, mask_i)
        intersection[i] = min(100, np.size(np.flatnonzero(
            mask_intersection)) * np.square(mask_res))

    if intersection.shape[0] == 1:
        intersection = intersection[0]

    return intersection
