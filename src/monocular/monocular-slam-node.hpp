#ifndef __MONOCULAR_SLAM_NODE_HPP__
#define __MONOCULAR_SLAM_NODE_HPP__

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include "geometry_msgs/msg/pose_stamped.hpp"

#include "System.h"
#include "Frame.h"
#include "Map.h"
#include "Tracking.h"

#include "utility.hpp"


class MonocularSlamNode : public rclcpp::Node
{
public:
    MonocularSlamNode(ORB_SLAM3::System* pSLAM);

    ~MonocularSlamNode();

private:
    using ImageMsg = sensor_msgs::msg::Image;

    void GrabImage(const sensor_msgs::msg::Image::SharedPtr msg);
    void GrabImu(const sensor_msgs::msg::Imu::SharedPtr msg);

    // --- 数据处理与发布 ---
    void PublishData(const Sophus::SE3f& Tcw, const rclcpp::Time& stamp);
    void PublishMapPoints();

    // 图片订阅
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_image_subscriber;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr m_imu_subscriber;

    // --- 发布者 ---
    
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr m_pose_publisher; // 2. 发布 Pose 给 Rviz 和你自己看
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr m_odom_publisher;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_cloud_publisher;
    std::unique_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster;

    // 地图发布
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_cloud_publisher;
    
    ORB_SLAM3::System* m_SLAM;

    cv_bridge::CvImagePtr m_cvImPtr;

    cv::Ptr<cv::CLAHE> m_clahe;

    // --- IMU 缓冲区与同步逻辑 ---
    std::mutex m_mutex_imu;
    std::vector<ORB_SLAM3::IMU::Point> m_imu_buffer;
    double m_last_image_time = -1.0;

    // 坐标系转换常量 (相机系 -> ROS 机器人系)
    // 根据 REP-103 标准进行轴映射
    Eigen::Matrix3f m_R_vis_ros;
};

#endif
