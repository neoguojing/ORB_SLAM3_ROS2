#include "monocular-slam-node.hpp"

#include<opencv2/core/core.hpp>

// IMU数据定义
// 字段,数值,单位,物理意义
// timestamp,1520504428069562682,ns (纳秒),采样时间戳（需除以 109 转为秒）
// w_RS_S_x/y/z,"-0.027, -0.090, 0.022",rad⋅s−1,角速度（陀螺仪数据）
// a_RS_S_x/y/z,"0.169, 0.237, 9.734",m⋅s−2,线加速度（加速度计数据）

using std::placeholders::_1;

MonocularSlamNode::MonocularSlamNode(ORB_SLAM3::System* pSLAM)
:   Node("ORB_SLAM3_ROS2")
{
    m_SLAM = pSLAM;
    // --- 1. 初始化 TF 广播器 ---
    m_tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // --- 2. 初始化发布者 ---
    // 使用默认的可靠通信 (QoS 10)
    m_pose_publisher = this->create_publisher<geometry_msgs::msg::PoseStamped>("~/pose", 10);
    m_odom_publisher = this->create_publisher<nav_msgs::msg::Odometry>("~/odom", 10);
    
    // 地图点云数据量大，建议使用较小的队列以减少延迟
    m_cloud_publisher = this->create_publisher<sensor_msgs::msg::PointCloud2>("~/map_points", 5);

    // --- 3. 初始化订阅者 ---
    // 图像和 IMU 建议使用传感器专用的 QoS (Best Effort / Sensor Data)
    auto sensor_qos = rclcpp::SensorDataQoS();

    m_image_subscriber = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/image_raw", sensor_qos,
        std::bind(&MonocularSlamNode::GrabImage, this, std::placeholders::_1));

    m_imu_subscriber = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu/data_raw", sensor_qos,
        std::bind(&MonocularSlamNode::GrabImu, this, std::placeholders::_1));

    // --- 4. 初始化 CLAHE 优化器 (之前讨论的优化) ---
    m_clahe = cv::createCLAHE(3.0, cv::Size(8, 8));

    RCLCPP_INFO(this->get_logger(), "ORB-SLAM3 节点初始化完成，等待传感器数据...");
}

MonocularSlamNode::~MonocularSlamNode()
{
    // Stop all threads
    m_SLAM->Shutdown();

    // Save camera trajectory
    m_SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
}

void MonocularSlamNode::GrabImu(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(m_mutex_imu);
    // 将 ROS IMU 消息转换为 ORB_SLAM3 的 IMU 点
    m_imu_buffer.push_back(ORB_SLAM3::IMU::Point(
        msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z,
        msg->angular_velocity.x,    msg->angular_velocity.y,    msg->angular_velocity.z,
        Utility::StampToSec(msg->header.stamp) // 需实现时间戳转换工具
    ));
}


void MonocularSlamNode::GrabImage(const sensor_msgs::msg::Image::SharedPtr msg)
{
    double t_image = Utility::StampToSec(msg->header.stamp);
    
    // 1. 提取缓冲区中所有早于当前图像时间的 IMU 数据
    std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
    {
        std::lock_guard<std::mutex> lock(m_mutex_imu);
        if (!m_imu_buffer.empty()) {
            auto it = m_imu_buffer.begin();
            while (it != m_imu_buffer.end() && it->t <= t_image) {
                vImuMeas.push_back(*it);
                it++;
            }
            // 删除已经传递给 SLAM 的旧数据，保留可能属于下一帧的数据
            // 注意：通常保留最后一个点作为下一帧的起点
            if (it != m_imu_buffer.begin()) {
                m_imu_buffer.erase(m_imu_buffer.begin(), it - 1);
            }
        }
    }

    // 2. 图像预处理 (CLAHE 等)
    // cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "mono8");
    cv::Mat im;
    try {
        im = cv_bridge::toCvCopy(msg, "mono8")->image;
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
    // 这里可以加入你 main 里的 clahe->apply(...)
    m_clahe->apply(im, im);

    // 3. 核心调用：单目惯性跟踪
    // 对应 main 中的 SLAM.TrackMonocular(im, tframe, vImuMeas)
    Sophus::SE3f Tcw = m_SLAM->TrackMonocular(im, t_image, vImuMeas);

    // 4. 检查状态并发布
    if (m_SLAM->GetTrackingState() == ORB_SLAM3::Tracking::OK) {
        PublishData(Tcw, msg->header.stamp);
        // 每隔几帧发布一次点云，减轻树莓派压力
        if (m_frame_count++ % 5 == 0) PublishMapPoints();
    }
}

void MonocularSlamNode::PublishData(const Sophus::SE3f& Tcw, const rclcpp::Time& stamp)
{
    // 1. 获取相机在 SLAM 世界系下的位姿 (Twc)
    Sophus::SE3f Twc = Tcw.inverse();
    Eigen::Vector3f p_cv = Twc.translation();
    Eigen::Matrix3f R_cv = Twc.rotationMatrix();

    // 2. 定义坐标轴映射矩阵 (视觉系 -> ROS 机器人系)
    // 根据你的要求映射: ROS_X=SLAM_Z, ROS_Y=-SLAM_X, ROS_Z=-SLAM_Y
    Eigen::Matrix3f R_v2r;
    R_v2r << 0, 0, 1,  // ROS_X 轴对应 SLAM 的 Z
            -1, 0, 0,  // ROS_Y 轴对应 SLAM 的 -X
             0,-1, 0;  // ROS_Z 轴对应 SLAM 的 -Y

    // 3. 变换平移向量
    Eigen::Vector3f p_ros = R_v2r * p_cv;

    // 4. 变换旋转矩阵
    // 公式: R_new = R_v2r * R_old * R_v2r.transpose()
    Eigen::Matrix3f R_ros = R_v2r * R_cv * R_v2r.transpose();
    Eigen::Quaternionf q_ros(R_ros);
    q_ros.normalize();

    // 5. 填充并发布 PoseStamped 消息
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = "map"; // 全局地图坐标系
    
    pose_msg.pose.position.x = p_ros.x();
    pose_msg.pose.position.y = p_ros.y();
    pose_msg.pose.position.z = p_ros.z();
    
    pose_msg.pose.orientation.x = q_ros.x();
    pose_msg.pose.orientation.y = q_ros.y();
    pose_msg.pose.orientation.z = q_ros.z();
    pose_msg.pose.orientation.w = q_ros.w();

    m_pose_publisher->publish(pose_msg);

    // 6. 广播 TF 变换 (map -> base_link)
    // 这是 Nav2 路径规划的基础
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header = pose_msg.header;
    tf_msg.child_frame_id = "base_link"; // 假设相机位于机器人中心
    
    tf_msg.transform.translation.x = p_ros.x();
    tf_msg.transform.translation.y = p_ros.y();
    tf_msg.transform.translation.z = p_ros.z();
    
    tf_msg.transform.rotation.x = q_ros.x();
    tf_msg.transform.rotation.y = q_ros.y();
    tf_msg.transform.rotation.z = q_ros.z();
    tf_msg.transform.rotation.w = q_ros.w();

    m_tf_broadcaster->sendTransform(tf_msg);

    // 7. (可选) 发布 Odometry 供局部规划器使用
    auto odom_msg = nav_msgs::msg::Odometry();
    odom_msg.header = pose_msg.header;
    odom_msg.child_frame_id = "base_link";
    odom_msg.pose.pose = pose_msg.pose;
    // 注意：单目 SLAM 很难提供准确的速度 twist，
    // 如果没有融合 IMU 计算出的速度，此处 twist 建议保持默认(0)
    m_odom_publisher->publish(odom_msg);
}


void MonocularSlamNode::PublishMapPoints()
{
    // 1. 获取所有地图点
    std::vector<ORB_SLAM3::MapPoint*> vpMapPoints = m_SLAM->GetTrackedMapPoints(); // 只获取当前看到的点，或者使用 GetAllMapPoints()
    if (vpMapPoints.empty()) return;

    auto cloud_msg = sensor_msgs::msg::PointCloud2();
    cloud_msg.header.stamp = this->now();
    cloud_msg.header.frame_id = "map"; // 地图点是在 map 坐标系下的

    // 2. 设置点云格式（x, y, z）
    sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(vpMapPoints.size());

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");

    for (auto pMP : vpMapPoints) {
        if (!pMP || pMP->isBad()) continue;

        Eigen::Vector3f pos = pMP->GetWorldPos();

        // 3. 坐标系转换 (OpenCV -> ROS)
        // ORB-SLAM3: x-right, y-down, z-forward
        // ROS: x-forward, y-left, z-up
        // 转换逻辑取决于你发布 TF 时的变换，通常如下：
        *iter_x = pos.z();  // ROS x = SLAM z
        *iter_y = -pos.x(); // ROS y = -SLAM x
        *iter_z = -pos.y(); // ROS z = -SLAM y

        ++iter_x; ++iter_y; ++iter_z;
    }

    m_cloud_publisher->publish(cloud_msg);
}
