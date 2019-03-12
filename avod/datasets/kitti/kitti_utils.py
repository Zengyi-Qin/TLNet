import os

import numpy as np

from avod.core import obj_utils

from avod.core.label_cluster_utils import LabelClusterUtils
from avod.core.mini_batch_utils import MiniBatchUtils


class KittiUtils(object):
    # Definition for difficulty levels
    # These values are from Kitti dataset
    # 0 - easy, 1 - medium, 2 - hard
    HEIGHT = (40, 25, 25)
    OCCLUSION = (0, 1, 2)
    TRUNCATION = (0.15, 0.3, 0.5)

    def __init__(self, dataset):

        self.dataset = dataset

        # Label Clusters
        self.label_cluster_utils = LabelClusterUtils(self.dataset)

        self.clusters, self.std_devs = [None, None]

        # BEV source from dataset config
        self.bev_source = self.dataset.bev_source

        # Parse config
        self.config = dataset.config.kitti_utils_config
        self.area_extents = np.reshape(self.config.area_extents, (3, 2))
        self.bev_extents = self.area_extents[[0, 2]]
        self.voxel_size = self.config.voxel_size
        self.anchor_strides = np.reshape(self.config.anchor_strides, (-1, 2))

        self._density_threshold = self.config.density_threshold

        # Mini Batch Utils
        self.mini_batch_utils = MiniBatchUtils(self.dataset)
        self._mini_batch_dir = self.mini_batch_utils.mini_batch_dir

        # Label Clusters
        self.clusters, self.std_devs = \
            self.label_cluster_utils.get_clusters()

    def class_str_to_index(self, class_str):
        """
        Converts an object class type string into a integer index

        Args:
            class_str: the object type (e.g. 'Car', 'Pedestrian', or 'Cyclist')

        Returns:
            The corresponding integer index for a class type, starting at 1
            (0 is reserved for the background class).
            Returns -1 if we don't care about that class type.
        """
        if class_str in self.dataset.classes:
            return self.dataset.classes.index(class_str) + 1

        raise ValueError('Invalid class string {}, not in {}'.format(
            class_str, self.dataset.classes))

    def create_slice_filter(self, point_cloud, area_extents,
                            ground_plane, ground_offset_dist, offset_dist):
        """ Creates a slice filter to take a slice of the point cloud between
            ground_offset_dist and offset_dist above the ground plane

        Args:
            point_cloud: Point cloud in the shape (3, N)
            area_extents: 3D area extents
            ground_plane: ground plane coefficients
            offset_dist: max distance above the ground
            ground_offset_dist: min distance above the ground plane

        Returns:
            A boolean mask if shape (N,) where
                True indicates the point should be kept
                False indicates the point should be removed
        """

        # Filter points within certain xyz range and offset from ground plane
        offset_filter = obj_utils.get_point_filter(point_cloud, area_extents,
                                                   ground_plane, offset_dist)

        # Filter points within 0.2m of the road plane
        road_filter = obj_utils.get_point_filter(point_cloud, area_extents,
                                                 ground_plane,
                                                 ground_offset_dist)

        slice_filter = np.logical_xor(offset_filter, road_filter)
        return slice_filter
   
    def get_anchors_info(self, classes_name, anchor_strides, sample_name):

        anchors_info = self.mini_batch_utils.get_anchors_info(classes_name,
                                                              anchor_strides,
                                                              sample_name)
        return anchors_info

    def get_ground_plane(self, sample_name):
        """Reads the ground plane for the sample

        Args:
            sample_name: name of the sample, e.g. '000123'

        Returns:
            ground_plane: ground plane coefficients
        """
        ground_plane = obj_utils.get_road_plane(int(sample_name),
                                                self.dataset.planes_dir)
        return ground_plane

    def filter_labels(self, objects,
                      classes=None,
                      difficulty=None,
                      max_occlusion=None):
        """Filters ground truth labels based on class, difficulty, and
        maximum occlusion

        Args:
            objects: A list of ground truth instances of Object Label
            classes: (optional) classes to filter by, if None
                all classes are used
            difficulty: (optional) KITTI difficulty rating as integer
            max_occlusion: (optional) maximum occlusion to filter objects

        Returns:
            filtered object label list
        """
        if classes is None:
            classes = self.dataset.classes

        objects = np.asanyarray(objects)
        filter_mask = np.ones(len(objects), dtype=np.bool)

        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]

            if filter_mask[obj_idx]:
                if not self._check_class(obj, classes):
                    filter_mask[obj_idx] = False
                    continue

            # Filter by difficulty (occlusion, truncation, and height)
            if difficulty is not None and \
                    not self._check_difficulty(obj, difficulty):
                filter_mask[obj_idx] = False
                continue

            if max_occlusion and \
                    obj.occlusion > max_occlusion:
                filter_mask[obj_idx] = False
                continue

        return objects[filter_mask]

    def _check_difficulty(self, obj, difficulty):
        """This filters an object by difficulty.
        Args:
            obj: An instance of ground-truth Object Label
            difficulty: An int defining the KITTI difficulty rate
        Returns: True or False depending on whether the object
            matches the difficulty criteria.
        """

        return ((obj.occlusion <= self.OCCLUSION[difficulty]) and
                (obj.truncation <= self.TRUNCATION[difficulty]) and
                (obj.y2 - obj.y1) >= self.HEIGHT[difficulty])

    def _check_class(self, obj, classes):
        """This filters an object by class.
        Args:
            obj: An instance of ground-truth Object Label
        Returns: True or False depending on whether the object
            matches the desired class.
        """
        return obj.type in classes
