#!/usr/bin/env python3

import rospy
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker
from detection_msgs.msg import BoundingBoxes
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import tf2_geometry_msgs
from collections import deque
import message_filters

class CameraLidarFusion:
    """
    Fuses camera (YOLOv10) and LiDAR detections to get:
    - Accurate 3D position (from LiDAR)
    - Cone color (from camera)
    """
    
    def __init__(self):
        rospy.init_node('camera_lidar_fusion', anonymous=True)
        
        # ========== PARAMETERS ==========
        self.fusion_distance_threshold = rospy.get_param('~fusion_threshold', 0.8)  # 80cm
        self.use_depth_map = rospy.get_param('~use_depth_map', True)
        
        # TF for transformations
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Storage for detections (with timeout)
        self.camera_detections = deque(maxlen=5)  # Last 5 camera frames
        self.lidar_detections = deque(maxlen=5)   # Last 5 LiDAR frames
        
        # ========== SUBSCRIBERS ==========
        # Using message_filters for time synchronization
        self.camera_sub = message_filters.Subscriber('/yolov10/detections', 
                                                     BoundingBoxes)
        self.lidar_sub = message_filters.Subscriber('/lidar_detections',
                                                    MarkerArray)
        
        # Optional: ZED depth/point cloud
        if self.use_depth_map:
            self.depth_sub = rospy.Subscriber('/zed/point_cloud/cloud_registered',
                                             PointCloud2,
                                             self.depth_callback,
                                             queue_size=1)
            self.depth_cloud = None
        
        # Time synchronizer (allow 0.1s difference)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.camera_sub, self.lidar_sub],
            queue_size=10,
            slop=0.1  # 100ms tolerance
        )
        self.ts.registerCallback(self.fusion_callback)
        
        # ========== PUBLISHERS ==========
        self.fused_pub = rospy.Publisher('/fused_cones',
                                         MarkerArray,
                                         queue_size=10)
        
        # Statistics
        self.fusion_count = 0
        self.camera_only_count = 0
        self.lidar_only_count = 0
        
        rospy.loginfo("Camera-LiDAR Fusion Node Ready!")
        rospy.loginfo(f"   - Fusion threshold: {self.fusion_distance_threshold}m")
        rospy.loginfo(f"   - Using depth map: {self.use_depth_map}")
    
    def depth_callback(self, cloud_msg):
        """Store latest depth point cloud from ZED"""
        self.depth_cloud = cloud_msg
    
    def fusion_callback(self, camera_msg, lidar_msg):
        """
        Main fusion logic:
        1. Transform all detections to base_link frame
        2. Match camera and LiDAR detections by proximity
        3. Create fused cone markers (position from LiDAR, color from camera)
        """
        
        # Extract camera detections with depth
        camera_cones = self.extract_camera_cones(camera_msg)
        
        # Extract LiDAR detections
        lidar_cones = self.extract_lidar_cones(lidar_msg)
        
        # Perform fusion
        fused_cones = self.fuse_detections(camera_cones, lidar_cones)
        
        # Publish results
        self.publish_fused_cones(fused_cones)
        
        # Statistics
        self.fusion_count = len(fused_cones)
        rospy.loginfo_throttle(2, 
            f"Fusion: {self.fusion_count} cones "
            f"(Cam: {len(camera_cones)}, LiDAR: {len(lidar_cones)})")
    
    def extract_camera_cones(self, bbox_msg):
        """
        Extract cone positions from camera detections.
        If depth is available, use it. Otherwise, estimate from image.
        """
        camera_cones = []
        
        for bbox in bbox_msg.bounding_boxes:
            # Calculate bounding box center
            cx = (bbox.xmin + bbox.xmax) / 2
            cy = (bbox.ymin + bbox.ymax) / 2
            
            # Get 3D position
            if self.use_depth_map and self.depth_cloud is not None:
                position_3d = self.get_depth_at_pixel(cx, cy)
            else:
                # Fallback: estimate distance from bbox size
                position_3d = self.estimate_distance_from_bbox(bbox)
            
            if position_3d is not None:
                camera_cones.append({
                    'position': position_3d,
                    'color': bbox.Class,
                    'confidence': bbox.probability,
                    'source': 'camera'
                })
        
        return camera_cones
    
    def get_depth_at_pixel(self, px, py):
        """Get 3D position from ZED depth cloud at pixel (px, py)"""
        if self.depth_cloud is None:
            return None
        
        try:
            # Point cloud is organized (has width and height)
            # Calculate point index
            px_int = int(px)
            py_int = int(py)
            
            # Get point from cloud
            points = list(pc2.read_points(self.depth_cloud,
                                         field_names=("x", "y", "z"),
                                         skip_nans=True,
                                         uvs=[(px_int, py_int)]))
            
            if len(points) > 0:
                point = points[0]
                
                # Transform from camera frame to base_link
                point_stamped = PointStamped()
                point_stamped.header = self.depth_cloud.header
                point_stamped.point.x = point[0]
                point_stamped.point.y = point[1]
                point_stamped.point.z = point[2]
                
                transformed = self.tf_buffer.transform(point_stamped,
                                                      "base_link",
                                                      rospy.Duration(0.1))
                
                return np.array([transformed.point.x,
                                transformed.point.y,
                                transformed.point.z])
        except Exception as e:
            rospy.logwarn_throttle(5, f"Depth lookup failed: {e}")
        
        return None
    
    def estimate_distance_from_bbox(self, bbox):
        """
        Estimate cone distance from bounding box size.
        Based on pinhole camera model.
        Formula: distance = (real_height * focal_length) / pixel_height
        """
        # ZED 2i parameters (adjust for your camera)
        focal_length = 700  # pixels (approximate for ZED 2i at 720p)
        real_cone_height = 0.325  # meters
        
        bbox_height = bbox.ymax - bbox.ymin
        
        if bbox_height > 0:
            distance = (real_cone_height * focal_length) / bbox_height
            
            # Calculate x, y from image center
            image_width = 1280  # ZED 2i default width
            image_height = 720  # ZED 2i default height
            
            cx = (bbox.xmin + bbox.xmax) / 2
            cy = (bbox.ymin + bbox.ymax) / 2
            
            # Angle from image center
            angle_x = np.arctan((cx - image_width/2) / focal_length)
            
            x = distance * np.cos(angle_x)
            y = distance * np.sin(angle_x)
            
            return np.array([x, y, 0.0])
        
        return None
    
    def extract_lidar_cones(self, marker_msg):
        """Extract cone positions from LiDAR markers"""
        lidar_cones = []
        
        for marker in marker_msg.markers:
            try:
                # Transform to base_link if needed
                point_stamped = PointStamped()
                point_stamped.header = marker.header
                point_stamped.point = marker.pose.position
                
                if marker.header.frame_id != "base_link":
                    transformed = self.tf_buffer.transform(point_stamped,
                                                          "base_link",
                                                          rospy.Duration(0.1))
                    position = np.array([transformed.point.x,
                                        transformed.point.y,
                                        transformed.point.z])
                else:
                    position = np.array([marker.pose.position.x,
                                        marker.pose.position.y,
                                        marker.pose.position.z])
                
                lidar_cones.append({
                    'position': position,
                    'color': 'unknown',  # LiDAR doesn't know color
                    'confidence': 0.9,   # LiDAR is generally reliable
                    'source': 'lidar'
                })
            
            except Exception as e:
                rospy.logwarn_throttle(5, f"LiDAR transform failed: {e}")
        
        return lidar_cones
    
    def fuse_detections(self, camera_cones, lidar_cones):
        """
        Match camera and LiDAR detections:
        - If match found: use LiDAR position + camera color
        - If camera only: use camera (less reliable position)
        - If LiDAR only: use LiDAR (no color info)
        """
        fused_cones = []
        matched_lidar = set()
        matched_camera = set()
        
        # Try to match each camera detection with LiDAR
        for i, cam_cone in enumerate(camera_cones):
            best_match = None
            min_distance = float('inf')
            best_lidar_idx = None
            
            for j, lidar_cone in enumerate(lidar_cones):
                if j in matched_lidar:
                    continue
                
                # Calculate 2D distance (ignore Z for matching)
                cam_pos_2d = cam_cone['position'][:2]
                lidar_pos_2d = lidar_cone['position'][:2]
                
                distance = np.linalg.norm(cam_pos_2d - lidar_pos_2d)
                
                if distance < min_distance and distance < self.fusion_distance_threshold:
                    min_distance = distance
                    best_match = lidar_cone
                    best_lidar_idx = j
            
            # Create fused detection
            if best_match is not None:
                # FUSION: LiDAR position + Camera color
                fused_cones.append({
                    'position': best_match['position'],  # Use accurate LiDAR position
                    'color': cam_cone['color'],          # Use camera color
                    'confidence': (cam_cone['confidence'] + best_match['confidence']) / 2,
                    'source': 'fused'
                })
                matched_lidar.add(best_lidar_idx)
                matched_camera.add(i)
            else:
                # Camera only (no LiDAR match)
                if cam_cone['confidence'] > 0.6:  # Only if confident
                    fused_cones.append(cam_cone)
                    matched_camera.add(i)
        
        # Add unmatched LiDAR detections (high confidence)
        for j, lidar_cone in enumerate(lidar_cones):
            if j not in matched_lidar:
                fused_cones.append(lidar_cone)
        
        return fused_cones
    
    def publish_fused_cones(self, fused_cones):
        """Publish fused cone markers"""
        marker_array = MarkerArray()
        
        for i, cone in enumerate(fused_cones):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = rospy.Time.now()
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = cone['position'][0]
            marker.pose.position.y = cone['position'][1]
            marker.pose.position.z = 0.325 / 2  # Half cone height
            marker.pose.orientation.w = 1.0
            
            # Size
            marker.scale.x = 0.228
            marker.scale.y = 0.228
            marker.scale.z = 0.325
            
            # Color based on source
            color_name = cone.get('color', 'unknown')
            
            if color_name == 'blue_cone':
                marker.color.r, marker.color.g, marker.color.b = 0.0, 0.0, 1.0
            elif color_name == 'yellow_cone':
                marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0
            elif color_name == 'orange_cone':
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0
            elif color_name == 'large_orange_cone':
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.3, 0.0
            else:  # unknown or lidar only
                marker.color.r, marker.color.g, marker.color.b = 0.5, 0.5, 0.5
            
            # Alpha based on source
            if cone['source'] == 'fused':
                marker.color.a = 1.0  # Fully opaque (best quality)
            else:
                marker.color.a = 0.7  # Semi-transparent
            
            marker.lifetime = rospy.Duration(0.5)
            marker_array.markers.append(marker)
        
        self.fused_pub.publish(marker_array)

if __name__ == '__main__':
    try:
        fusion = CameraLidarFusion()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Fusion node terminated.")
