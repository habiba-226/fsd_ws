#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from detection_msgs.msg import BoundingBox, BoundingBoxes
import numpy as np
from ultralytics import YOLO

class YOLOv10ConeDetector:
    def __init__(self):
        rospy.init_node('yolov10_cone_detector', anonymous=True)
        
        # Parameters
        self.model_path = rospy.get_param('~model_path', 
                                          '$(find fsd_perception)/models/yolov10/best.pt')
        self.conf_threshold = rospy.get_param('~confidence_threshold', 0.5)
        self.image_topic = rospy.get_param('~image_topic', '/zed/left/image_raw')
        self.use_compressed = rospy.get_param('~use_compressed', False)
        
        # Load YOLOv10 model
        rospy.loginfo(f"Loading YOLOv10 model from: {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
            self.model.conf = self.conf_threshold
            rospy.loginfo("✅ YOLOv10 model loaded successfully!")
        except Exception as e:
            rospy.logerr(f"Failed to load model: {e}")
            return
        
        # Class names (from your training)
        self.class_names = {
            0: 'blue_cone',
            1: 'large_orange_cone',
            2: 'orange_cone',
            3: 'unknown_cone',
            4: 'yellow_cone'
        }
        
        # Colors for visualization (BGR format)
        self.class_colors = {
            'blue_cone': (255, 0, 0),           # Blue
            'yellow_cone': (0, 255, 255),       # Yellow
            'orange_cone': (0, 165, 255),       # Orange
            'large_orange_cone': (0, 100, 255), # Dark Orange
            'unknown_cone': (128, 128, 128)     # Gray
        }
        
        self.bridge = CvBridge()
        
        # Subscribers
        if self.use_compressed:
            self.image_sub = rospy.Subscriber(self.image_topic + '/compressed',
                                             CompressedImage, 
                                             self.compressed_image_callback,
                                             queue_size=1)
        else:
            self.image_sub = rospy.Subscriber(self.image_topic,
                                             Image,
                                             self.image_callback,
                                             queue_size=1)
        
        # Publishers
        self.detection_pub = rospy.Publisher('/yolov10/detections',
                                             BoundingBoxes,
                                             queue_size=10)
        self.marker_pub = rospy.Publisher('/yolov10/markers',
                                          MarkerArray,
                                          queue_size=10)
        self.debug_image_pub = rospy.Publisher('/yolov10/debug_image',
                                               Image,
                                               queue_size=10)
        
        # Statistics
        self.detection_count = 0
        self.fps_counter = 0
        self.last_fps_time = rospy.Time.now()
        
        rospy.loginfo("YOLOv10 Cone Detector Ready!")
        rospy.loginfo(f"   - Confidence threshold: {self.conf_threshold}")
        rospy.loginfo(f"   - Subscribing to: {self.image_topic}")
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_image(cv_image, msg.header)
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
    
    def compressed_image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.process_image(cv_image, msg.header)
        except Exception as e:
            rospy.logerr(f"Compressed image error: {e}")
    
    def process_image(self, cv_image, header):
        """Main detection processing"""
        
        # Run YOLOv10 inference
        results = self.model(cv_image, verbose=False)[0]
        
        # Create messages
        bounding_boxes_msg = BoundingBoxes()
        bounding_boxes_msg.header = header
        bounding_boxes_msg.image_header = header
        
        marker_array = MarkerArray()
        
        # Process each detection
        detections = results.boxes.data.cpu().numpy()
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2, conf, cls = detection
            cls = int(cls)
            
            # Get class name
            class_name = self.class_names.get(cls, 'unknown')
            
            # Skip unknown cones if confidence is too low
            if class_name == 'unknown_cone' and conf < 0.6:
                continue
            
            # Create bounding box message
            bbox = BoundingBox()
            bbox.Class = class_name
            bbox.probability = float(conf)
            bbox.xmin = int(x1)
            bbox.ymin = int(y1)
            bbox.xmax = int(x2)
            bbox.ymax = int(y2)
            bounding_boxes_msg.bounding_boxes.append(bbox)
            
            # Create marker for visualization
            marker = self.create_marker(i, x1, y1, x2, y2, class_name, conf, header)
            marker_array.markers.append(marker)
            
            # Draw on debug image
            color = self.class_colors.get(class_name, (255, 255, 255))
            cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), 
                         color, 2)
            
            # Add label with confidence
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(cv_image, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add FPS counter to image
        self.fps_counter += 1
        current_time = rospy.Time.now()
        time_diff = (current_time - self.last_fps_time).to_sec()
        
        if time_diff >= 1.0:
            fps = self.fps_counter / time_diff
            self.fps_counter = 0
            self.last_fps_time = current_time
            
            cv2.putText(cv_image, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            rospy.loginfo_throttle(5, 
                f"Detections: {len(detections)} | FPS: {fps:.1f}")
        
        # Publish messages
        self.detection_pub.publish(bounding_boxes_msg)
        self.marker_pub.publish(marker_array)
        
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.debug_image_pub.publish(debug_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Failed to publish debug image: {e}")
    
    def create_marker(self, marker_id, x1, y1, x2, y2, class_name, conf, header):
        """Create RViz marker for visualization"""
        marker = Marker()
        marker.header = header
        marker.header.frame_id = "zed_camera_link"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # Calculate center of bounding box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Position (we'll need depth for accurate 3D, for now just image space)
        marker.pose.position.x = cx / 1000.0  # Scale down
        marker.pose.position.y = cy / 1000.0
        marker.pose.position.z = 1.0  # Placeholder depth
        marker.pose.orientation.w = 1.0
        
        # Size based on bounding box
        marker.scale.x = (x2 - x1) / 1000.0
        marker.scale.y = (y2 - y1) / 1000.0
        marker.scale.z = 0.325  # Standard cone height
        
        # Color based on class
        color_bgr = self.class_colors.get(class_name, (255, 255, 255))
        marker.color.b = color_bgr[0] / 255.0
        marker.color.g = color_bgr[1] / 255.0
        marker.color.r = color_bgr[2] / 255.0
        marker.color.a = 1.0
        
        marker.lifetime = rospy.Duration(0.5)
        
        return marker

if __name__ == '__main__':
    try:
        detector = YOLOv10ConeDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("YOLOv10 Detector node terminated.")
