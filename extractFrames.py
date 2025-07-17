import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        self.frame_count = 0
        self.save_count = 0

        self.frame_interval = 10  
        self.output_dir = "frames"
        os.makedirs(self.output_dir, exist_ok=True)
        self.get_logger().info(f"Saving frames at interval of {self.frame_interval} to {self.output_dir}")

    def listener_callback(self, msg):
        try:
            if self.frame_count % self.frame_interval == 0:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                filename = os.path.join(self.output_dir, f"frame_{self.save_count:04d}.png")
                cv2.imwrite(filename, cv_image)
                self.get_logger().info(f"Saved: {filename}")
                self.save_count += 1
            self.frame_count += 1
        except Exception as e:
            self.get_logger().error(f"Failed to save frame: {e}")

def main(args=None):
    rclpy.init(args=args)
    image_saver = ImageSaver()
    rclpy.spin(image_saver)
    image_saver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
