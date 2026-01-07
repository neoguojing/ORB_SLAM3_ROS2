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
#include "tf2/LinearMath/Transform.hpp"
#include "tf2/LinearMath/Quaternion.hpp"
#include "tf2_ros/transform_listener.hpp"
#include "tf2_ros/buffer.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"


#include "System.h"
#include "Frame.h"
#include "Map.h"
#include "Tracking.h"

#include "utility.hpp"
#include <deque>
#include <cmath>
#include <limits>
#include <algorithm>


class MonocularSlamNode : public rclcpp::Node
{
public:
    MonocularSlamNode(ORB_SLAM3::System* pSLAM, bool useIMU,
        bool pubMap2Odom = false,bool pubOdom = true,bool pubMap2Base = false,bool pubDebugImage=false,bool pubCloudPoint=true);

    ~MonocularSlamNode();

private:
    using ImageMsg = sensor_msgs::msg::Image;

    void ProcessImage(const cv::Mat& im, const rclcpp::Time& stamp);

    void GrabImage(const sensor_msgs::msg::Image::SharedPtr msg);
    void GrabCompressedImage(const sensor_msgs::msg::CompressedImage::SharedPtr msg);

    void GrabImu(const sensor_msgs::msg::Imu::SharedPtr msg);

    std::vector<ORB_SLAM3::IMU::Point> SyncImuData(double t_image);
    Sophus::SE3f ExecuteTracking(const cv::Mat& im, double t_image, const std::vector<ORB_SLAM3::IMU::Point>& vImu);
    bool ExtractMotionInfo(Eigen::Vector3f& v_world);
    // --- 数据处理与发布 ---
    void PublishImageData(const rclcpp::Time& stamp);
    void PublishMapPoints(const rclcpp::Time& stamp);
    Sophus::SE3f GetStaticTransformAsSophus(const std::string& target_frame);
    void HandleSlamOutput(const Sophus::SE3f& Tcw, const rclcpp::Time& stamp,const Eigen::Vector3f* v_world,const ORB_SLAM3::IMU::Point* lastPoint);

    void PublishMap2OdomTF(const Eigen::Vector3d& p_map_base,const Eigen::Quaterniond& q_map_base,const rclcpp::Time& stamp);
    void PublishOdm(const Eigen::Matrix3f& R_cv,
        const Eigen::Vector3f* v_world,
        const ORB_SLAM3::IMU::Point* lastPoint,
        const rclcpp::Time& stamp);
    void PublishPos(const Eigen::Vector3d& p_map_base,const Eigen::Quaterniond& q_map_base,const rclcpp::Time& stamp);
    

    // 图片订阅
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_image_subscriber;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr m_compress_image_subscriber;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr m_imu_subscriber;

    // --- 发布者 ---
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr m_pose_publisher; // 2. 发布 Pose 给 Rviz 和你自己看
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr m_odom_publisher;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_cloud_publisher;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr m_debug_img_publisher;

    //tf 广播
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    
    ORB_SLAM3::System* m_SLAM;

    cv::Ptr<cv::CLAHE> m_clahe;


    cv_bridge::CvImagePtr m_cvImPtr;

    // 坐标系转换常量 (相机系 -> ROS 机器人系)
    // 根据 REP-103 标准进行轴映射
    Eigen::Matrix3f m_R_vis_ros;
    // 图片帧计数
    int m_frame_count = 0;
    // 上一次跟踪时间
    double m_lost_start_time = -1.0;
    // 存储上一次 SLAM 跟踪的纯耗时（单位：秒）
    double m_last_elapsed = 0.0;

    std::unordered_map<std::string, Sophus::SE3f> static_tf_cache_;
    // --- IMU 缓冲区与同步逻辑 ---
    Sophus::SE3f m_Tbc;
    bool m_bTbcLoaded = false;
    bool m_useIMU = false;
    std::mutex m_mutex_imu;
    std::deque<ORB_SLAM3::IMU::Point> m_imu_buffer;
    int64_t m_last_imu_ns_ = -1;              // 上一个接收的 IMU 的 nanoseconds（用于去重 & 单调性）
    double m_max_buffer_seconds_ = 5.0;       // 缓冲最大保留秒数
    double m_min_time_increment_ = 1e-6;      // 最小时间增量，避免 dt==0
    double m_max_imu_dt_ = 0.02;              // 允许的最大 IMU 间隔 (s)（用于 gap 警告）
    double m_max_accel_magnitude_ = 200.0;    // 保护阈值（可选）

    // 发布控制
    bool m_pub_map_odom = false;
    bool m_pub_odom = true;
    bool m_pub_pos = false;
    bool m_pub_debug_image = false;
    bool m_pub_cloud_point = true;
};

#endif
