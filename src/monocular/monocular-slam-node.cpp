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
    m_debug_img_publisher = this->create_publisher<sensor_msgs::msg::Image>("~/debug_image", 10);

    // --- 3. 初始化订阅者 ---
    // 图像和 IMU 建议使用传感器专用的 QoS (Best Effort / Sensor Data)
    auto sensor_qos = rclcpp::SensorDataQoS();
    sensor_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE); 
    m_image_subscriber = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/image_raw", 10,
        std::bind(&MonocularSlamNode::GrabImage, this, std::placeholders::_1));
    
    m_compress_image_subscriber = this->create_subscription<sensor_msgs::msg::CompressedImage>(
        "/camera/image_raw/compressed", 10,
        std::bind(&MonocularSlamNode::GrabCompressedImage, this, std::placeholders::_1));

    m_imu_subscriber = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu/data_raw", 10,
        std::bind(&MonocularSlamNode::GrabImu, this, std::placeholders::_1));

    // --- 4. 初始化 CLAHE 优化器 (之前讨论的优化) ---
    m_clahe = cv::createCLAHE(3.0, cv::Size(8, 8));

    // 1. 初始化坐标变换矩阵 (m_R_vis_ros)
    // 根据上述映射逻辑填充矩阵
    m_R_vis_ros << 0, 0, 1,  // ROS X = CV Z
                  -1, 0, 0,  // ROS Y = -CV X
                   0,-1, 0;  // ROS Z = -CV Y

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

void MonocularSlamNode::GrabCompressedImage(const sensor_msgs::msg::CompressedImage::SharedPtr msg){
    try {
        // 解码压缩图像
        // ProcessImage(cv_bridge::toCvCopy(msg, "mono8")->image, msg->header.stamp);
        ProcessImage(cv_bridge::toCvCopy(msg, "rgb8")->image, msg->header.stamp);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
}

void MonocularSlamNode::GrabImage(const sensor_msgs::msg::Image::SharedPtr msg){
    try {
        // toCvShare 可以减少内存拷贝，提高效率
        // ProcessImage(cv_bridge::toCvShare(msg, "mono8")->image, msg->header.stamp);
        ProcessImage(cv_bridge::toCvShare(msg, "rgb8")->image, msg->header.stamp);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
}

void MonocularSlamNode::ProcessImage(const cv::Mat& im, const rclcpp::Time& stamp)
{
    // auto avg = cv::mean(im);
    // RCLCPP_INFO(this->get_logger(), "图像均值: %.2f", avg[0]);

    // 如果这条不打印，说明是之前的 QoS 不匹配或网络包过大丢失问题
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
        "收到图像消息! 时间戳: %.3f, 宽度: %d, 高度: %d", 
        Utility::StampToSec(stamp), im.cols, im.rows);

    double t_image = Utility::StampToSec(stamp);

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

    cv::Mat im_gray = im;

    // 1. 如果输入是彩色图，转换为灰度图 (ORB-SLAM3 核心要求)
    // if (im_gray.channels() == 3) {
    //     cv::cvtColor(im_gray, im_gray, cv::COLOR_BGR2GRAY);
    // }

    // 这里可以加入你 main 里的 clahe->apply(...)
    // m_clahe->apply(im_gray, im_gray);

    // 3. 核心调用：单目惯性跟踪
    // --- DEBUG 打印 4: SLAM 开始执行（记录耗时） ---
    auto t1 = std::chrono::high_resolution_clock::now();
    // 3. 核心调用：根据是否有 IMU 数据选择调用接口
    Sophus::SE3f Tcw;

    if (vImuMeas.empty()) {
        // 如果没有 IMU 数据，调用纯单目接口
        // 注意：如果你的系统初始化为 IMU_MONOCULAR，传空可能会报错或依然不就绪
        Tcw = m_SLAM->TrackMonocular(im_gray, t_image);
    } else {
        // 只有有数据时才调用惯性接口
        // Tcw = m_SLAM->TrackMonocular(im_gray, t_image, vImuMeas);
        Tcw = m_SLAM->TrackMonocular(im_gray, t_image);
    }

    if(Tcw.matrix().isZero()) {
        // 如果返回的是空位姿，说明这一帧在预处理阶段就被丢弃了
        RCLCPP_WARN(this->get_logger(), "SLAM 跟踪丢失: %d！", Tcw.matrix().isZero());
        return;
    }

    // 获取当前帧的特征点数量（需要包含对应的头文件）
    int nFeatures = m_SLAM->GetTrackedKeyPointsUn().size(); 
    RCLCPP_INFO_THROTTLE(this->get_logger(),*this->get_clock(), 1000, "当前帧提取到的特征点数: %d", nFeatures);

    auto t2 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

    // 4. 检查状态并发布
    int state = m_SLAM->GetTrackingState();
    
    // --- DEBUG 打印 5: SLAM 结果 ---
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
        "SLAM 跟踪状态: %d (9=OK), 耗时: %.4f 秒", state, elapsed);

    // 4. 检查状态并发布
    if (state == ORB_SLAM3::Tracking::OK) {
        PublishData(Tcw, stamp);
        // 每隔几帧发布一次点云，减轻树莓派压力
        if (m_frame_count++ % 5 == 0) {
            PublishMapPoints(stamp);
            PublishImageData(im_gray,stamp);
        }
    } else if (state == ORB_SLAM3::Tracking::LOST) {
        RCLCPP_WARN(this->get_logger(), "SLAM 跟踪丢失: %d！", state);
    } else if (state == ORB_SLAM3::Tracking::NOT_INITIALIZED) {
        // 单目 SLAM 必须先初始化
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
            "SLAM 尚未初始化:%d，请缓慢移动相机以产生视差...", state);
    }
}

void PublishImageData(const cv::Mat& im, const rclcpp::Time& stamp){
    // 1. 检查输入是否为彩色，如果不是则转为彩色（增强鲁棒性）
    cv::Mat imCopy;
    if (im.channels() == 1) {
        cv::cvtColor(im, imCopy, cv::COLOR_GRAY2RGB);
    } else {
        imCopy = im.clone(); // 直接使用 RGB/BGR 原图
    }

    // 3. 获取跟踪状态下的特征点数据
    // 注意：具体函数名可能因你的 ORB-SLAM3 封装版本而异（常见如 GetTrackedKeyPointsUn）
    std::vector<cv::KeyPoint> vKeys = m_SLAM->GetTrackedKeyPoints(); 
    std::vector<bool> vMatches = m_SLAM->GetTrackedMapPointsMask(); 

    // 4. 在彩色图上渲染特征点
    for(size_t i = 0; i < vKeys.size(); i++) {
        if(vMatches[i]) {
            // 跟踪成功的点：绘制绿色实心圆
            cv::circle(imCopy, vKeys[i].pt, 2, cv::Scalar(0, 255, 0), -1);
            // 可选：画一个小十字，更清晰
            // cv::drawMarker(imCopy, vKeys[i].pt, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 4);
        } else {
            // 提取但未匹配的点：绘制红色空心小圆
            cv::circle(imCopy, vKeys[i].pt, 1, cv::Scalar(0, 0, 255), 1);
        }
    }

    // 5. 转换并发布消息
    // 注意：如果你的输入是 RGB，这里用 "rgb8"；如果是 BGR（OpenCV 默认），用 "bgr8"
    auto debug_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "rgb8", imCopy).toImageMsg();
    debug_msg->header.stamp = stamp;
    debug_msg->header.frame_id = "camera_link"; // 对应你 TF 树中的相机坐标系名称

    m_debug_img_publisher->publish(*debug_msg);
}

void MonocularSlamNode::PublishData(const Sophus::SE3f& Tcw, const rclcpp::Time& stamp)
{
    // --- DEBUG 打印 1: 进入发布函数 ---
    // 使用 THROTTLE 避免日志刷屏，每 1 秒输出一次
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "SLAM Tracking OK: Publishing Pose and TF...");

    // 1. 获取相机在 SLAM 世界系下的位姿 (Twc)
    Sophus::SE3f Twc = Tcw.inverse();
    Eigen::Vector3f p_cv = Twc.translation();
    Eigen::Matrix3f R_cv = Twc.rotationMatrix();

    // 3. 变换平移向量
    Eigen::Vector3f p_ros = m_R_vis_ros * p_cv;

    // 4. 变换旋转矩阵
    // 公式: R_new = R_v2r * R_old * R_v2r.transpose()
    Eigen::Matrix3f R_ros = m_R_vis_ros * R_cv * m_R_vis_ros.transpose();
    Eigen::Quaternionf q_ros(R_ros);
    q_ros.normalize();

    // --- DEBUG 打印 2: 坐标数值校验 ---
    // 检查平移是否超出了合理范围（比如超过了 100米），有助于发现初始化失败的情况
    if (p_ros.norm() > 100.0) {
        RCLCPP_WARN(this->get_logger(), "Warning: Pose seems too large! Dist: %.2f m", p_ros.norm());
    }

    // 打印当前位姿 (ROS 坐标系)
    RCLCPP_DEBUG(this->get_logger(), "Pos: [x:%.2f, y:%.2f, z:%.2f] | Quat: [x:%.2f, y:%.2f, z:%.2f, w:%.2f]",
                 p_ros.x(), p_ros.y(), p_ros.z(), q_ros.x(), q_ros.y(), q_ros.z(), q_ros.w());

    // 5. 填充并发布 PoseStamped 消息(调试使用)
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

    // --- 6. 广播 TF 变换 (map -> odom) ---
    // 逻辑：SLAM 告诉系统，里程计原点在全局地图的什么位置
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = "map";       // 父节点：全局地图
    tf_msg.child_frame_id = "odom";       // 子节点：里程计原点

    tf_msg.transform.translation.x = p_ros.x();
    tf_msg.transform.translation.y = p_ros.y();
    tf_msg.transform.translation.z = p_ros.z();
    tf_msg.transform.rotation.x = q_ros.x();
    tf_msg.transform.rotation.y = q_ros.y();
    tf_msg.transform.rotation.z = q_ros.z();
    tf_msg.transform.rotation.w = q_ros.w();

    m_tf_broadcaster->sendTransform(tf_msg);

    // --- 7. 发布 Odometry 消息 (作为 EKF 的视觉里程计输入) ---
    auto odom_msg = nav_msgs::msg::Odometry();

    // 7.1 设置 Header
    odom_msg.header.stamp = stamp; 
    // 修改点：设为 "odom"。告诉 EKF 这是相对于里程计原点的位姿
    odom_msg.header.frame_id = "odom"; 
    // 被观测的对象依然是机器人本体
    odom_msg.child_frame_id = "base_link";

    // 7.2 填充位姿数据
    // 注意：如果你的 pose_msg.pose 是相对于 map 的，
    // 而你现在想发相对于 odom 的，逻辑上它们在初始时刻是重合的。
    odom_msg.pose.pose = pose_msg.pose;

    // 7.3 填充协方差 (重要：作为里程计输入，通常比作为 map 输入时设置得略大一点)
    for(int i=0; i<36; i++) {
        odom_msg.pose.covariance[i] = 0.0;
    }
    // 给 x, y, z 设置方差 (例如 0.01 米，给 EKF 留出融合 IMU 的空间)
    odom_msg.pose.covariance[0]  = 0.01; // x
    odom_msg.pose.covariance[7]  = 0.01; // y
    odom_msg.pose.covariance[14] = 0.01; // z
    // 旋转方差
    odom_msg.pose.covariance[21] = 0.05; // roll
    odom_msg.pose.covariance[28] = 0.05; // pitch
    odom_msg.pose.covariance[35] = 0.05; // yaw

    // 7.4 填充速度数据 (Twist)
    // 如果不计算速度，必须给速度协方差赋极大值
    for(int i=0; i<36; i++) {
        odom_msg.twist.covariance[i] = 0.0;
    }
    odom_msg.twist.covariance[0]  = 1.0e6; // 完全不信任此速度
    odom_msg.twist.covariance[7]  = 1.0e6;
    odom_msg.twist.covariance[35] = 1.0e6;

    // 发布数据
    m_odom_publisher->publish(odom_msg);

    // --- DEBUG 打印 3: 循环计数 ---
    static int pub_count = 0;
    pub_count++;
    if(pub_count % 10 == 0) {
        RCLCPP_INFO(this->get_logger(), "Successfully published %d poses.", pub_count);
    }
}


void MonocularSlamNode::PublishMapPoints(const rclcpp::Time& stamp)
{
    // 1. 获取所有地图点
    std::vector<ORB_SLAM3::MapPoint*> vpMapPoints = m_SLAM->GetTrackedMapPoints(); // 只获取当前看到的点，或者使用 GetAllMapPoints()
    if (vpMapPoints.empty()) return;

    auto cloud_msg = sensor_msgs::msg::PointCloud2();
    cloud_msg.header.stamp = stamp;
    cloud_msg.header.frame_id = "map"; // 地图点是在 map 坐标系下的

    sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
    modifier.setPointCloud2FieldsByString(1, "xyz");

    // 预过滤有效点
    std::vector<Eigen::Vector3f> valid_points;
    for (auto pMP : vpMapPoints) {
        if (pMP && !pMP->isBad()) {
            // 使用与 Pose 统一的转换矩阵
            // 可选：高度过滤（假设 z 是高度）
            // if (pos_ros.z() < 0.05 || pos_ros.z() > 2.0) continue;
            valid_points.push_back(m_R_vis_ros * pMP->GetWorldPos());
        }
    }

    modifier.resize(valid_points.size());
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");

    for (const auto& pos : valid_points) {
        *iter_x = pos.x();
        *iter_y = pos.y();
        *iter_z = pos.z();
        ++iter_x; ++iter_y; ++iter_z;
    }

    m_cloud_publisher->publish(cloud_msg);
}
