# ELE 456 Project
# By: Nathan Kaye, Patrick Feliz, & Gianni Smith
import rclpy
from rclpy.node import Node
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time

class CylinderExplorer(Node):
    def __init__(self):
        super().__init__('cylinder_explorer')

        # Initializing the subscribers and publisher
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 1)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 1)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)

        # Setting the Parameters
        self.robot_position = [0.0, 0.0]
        # Set the starting position
        self.starting_position = [0.0, 0.0]
        self.robot_yaw = 0.0
        self.cylinders_detected = []
        self.visited_cylinders = set()
        self.target_cylinder = None
        self.scan_data = None
        self.zigzag_targets = self.define_zigzag_targets()
        self.zigzag_index = 0
        self.repulsive_force = np.array([0.0, 0.0])

        # Constants 
        self.max_linear_speed =  0.6 # max linear speed 
        self.max_angular_speed = 6.8 # creased max angular speed (from 2.0)
        self.target_tolerance = 0.5  # Reduced target tolerance (from 0.8)
        self.scan_cluster_tolerance = 0.2
        self.update_interval = 0.05  # Increase update frequency (remains the same)
        self.safety_distance = 0.3

        # Start control loop
        self.create_timer(self.update_interval, self.control_callback)
        self.start_time = time.time()

    def define_zigzag_targets(self):
        half_side = 5.0 
        spacing = 2.0  
        zigzag_targets = []

        for i in range(int(half_side * 2 / spacing)):
            x = -half_side + i * spacing
            y = half_side if i % 2 == 0 else -half_side
            zigzag_targets.append((x, y))
        return zigzag_targets

    def odom_callback(self, msg):
        self.robot_position = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        orientation = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        self.scan_data = msg  # Ensuring that the scan data is updated
        # Process LaserScan data to detect cylinders
        points = []
        repulsive_force = np.array([0.0, 0.0])
        angle = msg.angle_min
        for r in msg.ranges:
            if 0.1 < r < msg.range_max:
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                points.append([x, y])
                # Calculate repulsive force
                if r < self.safety_distance:
                    force_magnitude = 1.0 / (r**2)
                    force_angle = angle + self.robot_yaw
                    repulsive_force += force_magnitude * np.array([math.cos(force_angle), math.sin(force_angle)])
            angle += msg.angle_increment

        # Convert points to world coordinates
        world_points = []
        for x, y in points:
            x_world = self.robot_position[0] + x * math.cos(self.robot_yaw) - y * math.sin(self.robot_yaw)
            y_world = self.robot_position[1] + x * math.sin(self.robot_yaw) + y * math.cos(self.robot_yaw)
            world_points.append([x_world, y_world])

        # Cluster points into cylinders
        self.cylinders_detected = self.cluster_points(world_points)
        self.repulsive_force = repulsive_force

    def cluster_points(self, points):
        clusters = []
        for point in points:
            added = False
            for cluster in clusters:
                if np.linalg.norm(np.array(cluster) - np.array(point)) < self.scan_cluster_tolerance:
                    cluster.append(point)
                    added = True
                    break
            if not added:
                clusters.append([point])

        # Return centroids of clusters
        return [tuple(np.mean(cluster, axis=0)) for cluster in clusters]

    def control_callback(self):
        # Check if 90 seconds have passed
        if time.time() - self.start_time > 120:
            self.stop_robot()
            return

        # Check for unvisited cylinders
        unvisited = [c for c in self.cylinders_detected if not self.is_visited(c)]

        if unvisited:
            # Navigate to the nearest unvisited cylinder
            if not self.target_cylinder or self.is_target_reached(self.target_cylinder):
                if self.target_cylinder:
                    self.visited_cylinders.add(self.target_cylinder)
                unvisited.sort(key=lambda c: np.linalg.norm(np.array(self.robot_position) - np.array(c)))
                self.target_cylinder = unvisited[0]
                self.get_logger().info(f"New target: {self.target_cylinder}")
            self.navigate_to_target(self.target_cylinder)
        else:
            # If no targets detected, explore in a zig-zag pattern
            self.explore_zigzag()

    def explore_zigzag(self):
        target = self.zigzag_targets[self.zigzag_index]

        # If target reached, switch to the next zigzag point
        if self.is_target_reached(target):
            self.zigzag_index = (self.zigzag_index + 1) % len(self.zigzag_targets)
            self.get_logger().info(f"Switching to next zigzag point: {self.zigzag_targets[self.zigzag_index]}")
            target = self.zigzag_targets[self.zigzag_index]

        self.get_logger().info(f"Exploring in zigzag: {target}")
        self.navigate_to_target(target)

    def is_visited(self, cylinder, tolerance=0.3):  # Reduced tolerance
        for visited in self.visited_cylinders:
            if np.linalg.norm(np.array(cylinder) - np.array(visited)) < tolerance:
                return True
        return False

    def is_target_reached(self, target):
        if target is None:
            return False
        distance = np.linalg.norm(np.array(self.robot_position) - np.array(target))
        return distance < self.target_tolerance

    def navigate_to_target(self, target):
        dx = target[0] - self.robot_position[0]
        dy = target[1] - self.robot_position[1]
        distance = math.sqrt(dx**2 + dy**2)
        angle_to_target = math.atan2(dy, dx)

        angular_error = (angle_to_target - self.robot_yaw + math.pi) % (2 * math.pi) - math.pi

        # Calculate resulting force
        attractive_force = np.array([dx, dy])
        total_force = attractive_force - self.repulsive_force

        total_distance = np.linalg.norm(total_force)
        total_angle = math.atan2(total_force[1], total_force[0])

        angular_error = (total_angle - self.robot_yaw + math.pi) % (2 * math.pi) - math.pi

        # Adjust speed based on proximity to the target and obstacles
        current_speed = min(self.max_linear_speed, 0.5 * total_distance)
        adjusted_safety_distance = max(self.safety_distance, current_speed * 0.5)

        # Collision avoidance using scan data
        if self.scan_data:
            min_distance = min(self.scan_data.ranges)
            if min_distance < adjusted_safety_distance:
                self.get_logger().info("Avoiding obstacle")
                angular_error += math.pi / 4  # Turn away from the obstacle

        # Create movement command
        twist = Twist()
        twist.linear.x = current_speed
        twist.angular.z = max(-self.max_angular_speed, min(self.max_angular_speed, 2.0 * angular_error))

        # Publish movement
        self.cmd_vel_pub.publish(twist)

    def stop_robot(self):
        # Stop the robot
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    explorer = CylinderExplorer()
    rclpy.spin(explorer)
    explorer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
