#ifndef __MONOCULAR_SLAM_NODE_HPP__
#define __MONOCULAR_SLAM_NODE_HPP__

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <sensor_msgs/msg/compressed_image.hpp>
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include <cv_bridge/cv_bridge.hpp>
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include "tf2_ros/transform_broadcaster.hpp"

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

    void ProcessImage(const cv::Mat& im, const rclcpp::Time& stamp);

    void GrabImage(const sensor_msgs::msg::Image::SharedPtr msg);
    void GrabCompressedImage(const sensor_msgs::msg::CompressedImage::SharedPtr msg);

    void GrabImu(const sensor_msgs::msg::Imu::SharedPtr msg);

    // --- 数据处理与发布 ---
    void PublishData(const Sophus::SE3f& Tcw,const Eigen::Vector3f* v_world,const ORB_SLAM3::IMU::Point* lastPoint, const rclcpp::Time& stamp);
    void PublishImageData(const rclcpp::Time& stamp);

    void PublishMapPoints(const rclcpp::Time& stamp);

    // 图片订阅
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_image_subscriber;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr m_compress_image_subscriber;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr m_imu_subscriber;

    // --- 发布者 ---
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr m_pose_publisher; // 2. 发布 Pose 给 Rviz 和你自己看
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr m_odom_publisher;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_cloud_publisher;
    std::unique_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr m_debug_img_publisher;

    
    ORB_SLAM3::System* m_SLAM;

    cv::Ptr<cv::CLAHE> m_clahe;


    cv_bridge::CvImagePtr m_cvImPtr;
    // --- IMU 缓冲区与同步逻辑 ---
    std::mutex m_mutex_imu;
    std::vector<ORB_SLAM3::IMU::Point> m_imu_buffer;
    // 坐标系转换常量 (相机系 -> ROS 机器人系)
    // 根据 REP-103 标准进行轴映射
    Eigen::Matrix3f m_R_vis_ros;
    int m_frame_count = 0;
};

#endif
