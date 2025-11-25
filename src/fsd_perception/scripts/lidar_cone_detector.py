#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
from sklearn.cluster import DBSCAN
import tf2_ros
import tf2_geometry_msgs

class LidarConeDetector:
    """
    Improved LiDAR cone detection based on:
    - DBSCAN clustering
    - Geometric filtering (cone-like shape)
    - Size filtering (cone dimensions)
    """
    
    def __init__(self):
        rospy.init_node('lidar_cone_detector', anonymous=True)
        
        # ========== PARAMETERS ==========
        # Clustering
        self.cluster_tolerance = rospy.get_param('~cluster_tolerance', 0.3)  # 30cm
        self.min_cluster_size = rospy.get_param('~min_cluster_size', 3)
        self.max_cluster_size = rospy.get_param('~max_cluster_size', 50)
        
        # Range filtering
        self.min_range = rospy.get_param('~min_range', 0.5)   # 0.5m minimum
        self.max_range = rospy.get_param('~max_range', 15.0)  # 15m maximum
        
        # Cone geometry constraints (Formula Student regulations)
        self.cone_diameter = 0.228  # meters (regulations: 228mm)
        self.cone_height = 0.325    # meters (regulations: 325mm)
        self.diameter_tolerance = 0.15  # ±15cm tolerance
        
        # Height filtering (2D LiDAR height from ground)
        self.lidar_height = rospy.get_param('~lidar_height', 0.3)  # 30cm from ground
        
        # TF for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # ========== SUBSCRIBERS ==========
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, 
                                         self.scan_callback, queue_size=1)
        
        # ========== PUBLISHERS ==========
        self.marker_pub = rospy.Publisher('/lidar_detections', 
                                          MarkerArray, queue_size=10)
        self.pointcloud_pub = rospy.Publisher('/lidar_clusters',
                                              PointCloud2, queue_size=10)
        
        # ========== STATISTICS ==========
        self.detection_count = 0
        self.frame_count = 0
        
        rospy.loginfo("LiDAR Cone Detector Ready!")
        rospy.loginfo(f"   - Cluster tolerance: {self.cluster_tolerance}m")
        rospy.loginfo(f"   - Detection range: {self.min_range}-{self.max_range}m")
        rospy.loginfo(f"   - Cone diameter: {self.cone_diameter}m ±{self.diameter_tolerance}m")
    
    def scan_callback(self, scan_msg):
        """Process LaserScan data"""
        
        # Convert scan to cartesian points
        points = self.scan_to_cartesian(scan_msg)
        
        if len(points) < self.min_cluster_size:
            return
        
        # Cluster points using DBSCAN
        clusters = self.cluster_points(points)
        
        # Filter clusters by geometry
        valid_cones = self.filter_by_geometry(clusters)
        
        # Publish markers
        self.publish_markers(valid_cones, scan_msg.header)
        
        # Statistics
        self.frame_count += 1
        self.detection_count = len(valid_cones)
        
        if self.frame_count % 10 == 0:
            rospy.loginfo(f"LiDAR: {self.detection_count} cones detected")
    
    def scan_to_cartesian(self, scan):
        """Convert LaserScan to cartesian coordinates with filtering"""
        points = []
        angles = np.arange(scan.angle_min, 
                          scan.angle_max + scan.angle_increment,
                          scan.angle_increment)
        
        for i, (angle, distance) in enumerate(zip(angles, scan.ranges)):
            # Range filtering
            if not (self.min_range < distance < self.max_range):
                continue
            
            # Ignore NaN/Inf values
            if not np.isfinite(distance):
                continue
            
            # Convert to cartesian
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            
            points.append([x, y])
        
        return np.array(points)
    
    def cluster_points(self, points):
        """Cluster points using DBSCAN algorithm"""
        if len(points) < self.min_cluster_size:
            return []
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=self.cluster_tolerance,
                           min_samples=self.min_cluster_size)
        labels = clustering.fit_predict(points)
        
        # Group points by cluster
        clusters = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            
            cluster_points = points[labels == label]
            
            # Check cluster size
            if self.min_cluster_size <= len(cluster_points) <= self.max_cluster_size:
                clusters.append(cluster_points)
        
        return clusters
    
    def filter_by_geometry(self, clusters):
        """
        Filter clusters based on cone geometry:
        - Approximate circular shape
        - Correct diameter
        - Reasonable spread
        """
        valid_cones = []
        
        for cluster in clusters:
            # Calculate cluster properties
            centroid = cluster.mean(axis=0)
            distances = np.linalg.norm(cluster - centroid, axis=1)
            
            # Estimate diameter (max spread)
            diameter = 2 * distances.max()
            
            # Check if diameter matches cone size
            expected_diameter = self.cone_diameter
            diameter_diff = abs(diameter - expected_diameter)
            
            if diameter_diff < self.diameter_tolerance:
                # Calculate cluster compactness (should be circular)
                std_dev = distances.std()
                
                # Cones should be relatively compact
                if std_dev < 0.15:  # 15cm standard deviation
                    valid_cones.append({
                        'centroid': centroid,
                        'points': cluster,
                        'diameter': diameter,
                        'confidence': 1.0 - (diameter_diff / self.diameter_tolerance)
                    })
        
        return valid_cones
    
    def publish_markers(self, cones, header):
        """Publish RViz markers for detected cones"""
        marker_array = MarkerArray()
        
        for i, cone in enumerate(cones):
            marker = Marker()
            marker.header = header
            marker.header.frame_id = "laser"  # or "lidar_link"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = cone['centroid'][0]
            marker.pose.position.y = cone['centroid'][1]
            marker.pose.position.z = self.cone_height / 2  # Center of cone
            marker.pose.orientation.w = 1.0
            
            # Size
            marker.scale.x = self.cone_diameter
            marker.scale.y = self.cone_diameter
            marker.scale.z = self.cone_height
            
            # Color (orange - we don't know color from LiDAR alone)
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            marker.color.a = 0.8  # Semi-transparent
            
            # Lifetime
            marker.lifetime = rospy.Duration(0.5)
            
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)

if __name__ == '__main__':
    try:
        detector = LidarConeDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("LiDAR Detector node terminated.")
