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
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    // --- 2. 初始化发布者 ---
    // 使用默认的可靠通信 (QoS 10)
    m_pose_publisher = this->create_publisher<geometry_msgs::msg::PoseStamped>("/slam3/pose", 10);
    m_odom_publisher = this->create_publisher<nav_msgs::msg::Odometry>("/slam3/odom", 10);
    
    // 地图点云数据量大，建议使用较小的队列以减少延迟
    m_cloud_publisher = this->create_publisher<sensor_msgs::msg::PointCloud2>("/slam3/map_points", 5);
    m_debug_img_publisher = this->create_publisher<sensor_msgs::msg::Image>("/slam3/debug_image", 10);

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
    // OpenCV (x-right, y-down, z-forward) -> ROS (x-forward, y-left, z-up)
    // 这个矩阵也是它自身的逆 (R.transpose() == R)
    m_R_vis_ros << 0, 0, 1,  // ROS X = CV Z
                  -1, 0, 0,  // ROS Y = -CV X
                   0,-1, 0;  // ROS Z = -CV Y

    RCLCPP_INFO(this->get_logger(), "ORB-SLAM3 节点初始化完成，等待传感器数据...");
}

MonocularSlamNode::MonocularSlamNode(ORB_SLAM3::System* pSLAM, bool useIMU)
:  MonocularSlamNode(pSLAM)
{
    m_useIMU = useIMU;
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
    // 尝试获取Tbc
    if (!m_bTbcLoaded && m_SLAM->GetSetting()) {
        m_Tbc = m_SLAM->GetSetting()->Tbc();
        m_bTbcLoaded = true;
        Utility::PrintSophusSE3("Tbc 外参", m_Tbc);
    }

    double t_image = Utility::StampToSec(stamp);

    // 1. 提取缓冲区中所有早于当前图像时间的 IMU 数据
    // 1. 增加频率监控
    static double last_t_imu = -1;
    double current_t_imu = Utility::StampToSec(stamp);
    if(last_t_imu > 0 && (current_t_imu - last_t_imu) > 0.05) {
        RCLCPP_WARN(this->get_logger(), "IMU 数据丢包严重! dt: %f", current_t_imu - last_t_imu);
    }
    last_t_imu = current_t_imu;

    std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
    {
        std::lock_guard<std::mutex> lock(m_mutex_imu);
        if (!m_imu_buffer.empty()) {
            // 关键：确保 IMU 数据已经覆盖了图像时间戳
            if (m_imu_buffer.back().t < t_image) {
                RCLCPP_INFO(this->get_logger(), "等待 IMU 数据到达当前图像时间...");
                return; 
            }
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

    // 3. 检查 IMU 数据的数值
    // 确保 IMU 时间戳严格递增
    static double last_imu_t = -1.0;
    std::vector<ORB_SLAM3::IMU::Point> vFilteredImu;

    for(auto &p : vImuMeas) {
        if (p.t <= last_imu_t) continue; // 跳过时间戳倒退或重复的点
        
        // 拦截极其离谱的 dt (例如 > 0.1s 甚至更大)
        if (last_imu_t > 0 && (p.t - last_imu_t) > 0.1) {
            RCLCPP_WARN(this->get_logger(), "IMU 数据断层! dt: %f", p.t - last_imu_t);
        }
        
        vFilteredImu.push_back(p);
        last_imu_t = p.t;
    }
    RCLCPP_INFO_THROTTLE(this->get_logger(),*this->get_clock(), 1000, "当前IMU个数: %d，合法IMU个数: %d", (int)vImuMeas.size(), (int)vFilteredImu.size());

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
    // 参考系：slam
    Sophus::SE3f Tcw; 
    if (!m_useIMU) {
        // 如果没有 IMU 数据，调用纯单目接口
        // 注意：如果你的系统初始化为 IMU_MONOCULAR，传空可能会报错或依然不就绪
        Tcw = m_SLAM->TrackMonocular(im_gray, t_image);
    } else {
        // 只有有数据时才调用惯性接口
        if (vFilteredImu.empty()){
            RCLCPP_WARN(this->get_logger(), "警告: IMU 数据为空，但预期不应如此！");
            return;
        } else {
            Tcw = m_SLAM->TrackMonocular(im_gray, t_image, vFilteredImu);
        }
    }

    if(Tcw.matrix().isZero() || Tcw.matrix().array().isNaN().any() || Tcw.matrix().array().isInf().any()) {
        // 如果返回的是空位姿，说明这一帧在预处理阶段就被丢弃了
        std::stringstream ss;
        // 使用 Eigen 的内置格式化输出矩阵
        ss << "\n" << Tcw.matrix();
        RCLCPP_ERROR(this->get_logger(), 
            "--- 检测到无效位姿 (NaN/Inf) ---\n"
            "Tcw Matrix: %s\n"
            "--------------------------------", 
            ss.str().c_str());
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
        // 1. 获取 SLAM 世界系下的速度
        Eigen::Vector3f v_world;
        const Eigen::Vector3f* v_world_ptr = nullptr;
        const ORB_SLAM3::IMU::Point* imu_ptr = nullptr;

        // 获取 Tracker 指针
        ORB_SLAM3::Tracking* pTracker = m_SLAM->GetTracker();
        if(pTracker) {
            // 关键：不要拷贝 Frame，只拷贝 Vector3f（这是基本类型，拷贝极快且相对安全）
            // 最好判断一下当前的模式是否支持速度获取
            try {
                v_world = pTracker->mCurrentFrame.GetVelocity();
                Utility::PrintVector3f("v_world", v_world);
                if(!v_world.isZero() && !v_world.array().isNaN().any() && v_world.norm() < 100.0f ) {
                    v_world_ptr = &v_world;
                } else {
                    RCLCPP_WARN(this->get_logger(), "SLAM速度异常（过大或NaN），已拦截: [%f]", v_world.norm());
                    v_world.setZero(); // 强制归零，防止污染 EKF
                }
                
            } catch (...) {
                RCLCPP_ERROR(this->get_logger(), "读取速度失败");
            }
        }
    
        // 2. IMU：确保队列非空，参考系：base_link
        if (!vFilteredImu.empty())
        {
            imu_ptr = &vFilteredImu.back();
        }

        PublishData(Tcw, v_world_ptr, imu_ptr, stamp);

        // 每隔几帧发布一次点云，减轻树莓派压力
        if (m_frame_count++ % 5 == 0) {
            PublishMapPoints(stamp);
            PublishImageData(stamp);
        }
    } else if (state == ORB_SLAM3::Tracking::LOST) {
        RCLCPP_WARN(this->get_logger(), "SLAM 跟踪丢失: %d！", state);
    } else if (state == ORB_SLAM3::Tracking::NOT_INITIALIZED) {
        // 单目 SLAM 必须先初始化
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
            "SLAM 尚未初始化:%d，请缓慢移动相机以产生视差...", state);
    }
}

void MonocularSlamNode::PublishImageData(const rclcpp::Time& stamp){
     // 3. 获取跟踪状态下的特征点数据
    cv::Mat bgr = m_SLAM->GetFrameDrawer()->DrawFrame();
    if (bgr.empty()) return;

    cv::Mat rgb;
    // 5. 转换并发布消息
    // 注意：如果你的输入是 RGB，这里用 "rgb8"；如果是 BGR（OpenCV 默认），用 "bgr8"
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    auto debug_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "rgb8", rgb).toImageMsg();
    debug_msg->header.stamp = stamp;
    debug_msg->header.frame_id = "camera_link"; // 对应你 TF 树中的相机坐标系名称

    m_debug_img_publisher->publish(*debug_msg);
}

/**
 * @brief 发布 map -> odom 变换
 * @param p_map_base SLAM输出的平移 (Map -> imu_link, 此时应已完成ROS轴转换)
 * @param q_map_base SLAM输出的旋转 (Map -> imu_link, 此时应已完成ROS轴转换)
 * @param stamp      图像帧对应的时间戳
 */
void MonocularSlamNode::PublishMap2OdomTF(
    const Eigen::Vector3d& p_map_base,
    const Eigen::Quaterniond& q_map_base,
    const rclcpp::Time& stamp,
    bool is_imu=true) 
{
    try {
        // 1. 将输入的位姿包装为 tf2 格式
        // 注意：tf2 使用 double 精度。如果传入的是 Eigen::Vector3f，请确保已 cast<double>()
        tf2::Transform map_to_sensor;
        map_to_sensor.setOrigin(tf2::Vector3(p_map_base.x(), p_map_base.y(), p_map_base.z()));
        map_to_sensor.setRotation(tf2::Quaternion(
            q_map_base.x(), q_map_base.y(), q_map_base.z(), q_map_base.w()));

        // 2. 动态对齐到 base_link
        // 语义：确定 SLAM 算的到底是哪一部分，然后查表转到底盘中心
        std::string sensor_frame = is_imu ? "imu_link" : "camera_link_optical";
        
        // 获取 base_link -> sensor 的静态外参 (由 URDF 提供)
        auto base_to_sensor_msg = tf_buffer_->lookupTransform("base_link", sensor_frame, tf2::TimePointZero);
        tf2::Transform base_to_sensor;
        tf2::fromMsg(base_to_sensor_msg.transform, base_to_sensor);

        // T_map_base = T_map_sensor * T_base_sensor^-1
        tf2::Transform map_to_base = map_to_sensor * base_to_sensor.inverse();

        // 3. 获取里程计位姿 (odom -> base_link)
        // 查找 stamp 时刻位姿，同步 SLAM 延迟
        auto odom_to_base_msg = tf_buffer_->lookupTransform("odom", "base_link", stamp, rclcpp::Duration::from_seconds(0.05));
        tf2::Transform odom_to_base;
        tf2::fromMsg(odom_to_base_msg.transform, odom_to_base);

        // 4. 反算补偏量: T_map_odom = T_map_base * T_odom_base^-1
        tf2::Transform map_to_odom = map_to_base * odom_to_base.inverse();

        // 5. 发布变换
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = stamp;
        tf_msg.header.frame_id = "map";
        tf_msg.child_frame_id = "odom";
        tf_msg.transform = tf2::toMsg(map_to_odom);

        tf_broadcaster_->sendTransform(tf_msg);

    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "TF Sync Error: %s", ex.what());
    }
}

void MonocularSlamNode::PublishData(const Sophus::SE3f& Tcw,const Eigen::Vector3f* v_world,const ORB_SLAM3::IMU::Point* lastPoint, const rclcpp::Time& stamp)
{
    // --- DEBUG 打印 1: 进入发布函数 ---
    // 使用 THROTTLE 避免日志刷屏，每 1 秒输出一次
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "SLAM Tracking OK: Publishing Pose");

    // 1. 获取外参 Tbc (Camera to Body)
    // 建议：直接从 settings 获取对象，不要直接取地址
    Sophus::SE3f* pTbc = nullptr;
    // 检查 Tbc 是否有效 (不是单位阵且不为零)
    // 注：Sophus 没有 isZero()，通常检查其 matrix().isIdentity()
    if (m_bTbcLoaded) {
        pTbc = &m_Tbc;
    }
    
    Eigen::Matrix3f R_cv;
    Eigen::Vector3f p_ros;
    Eigen::Quaternionf q_ros;
    Utility::ConvertSLAMPoseToROS(Tcw,R_cv,p_ros,q_ros,pTbc);

    // 3. 检查平移向量 p_ros (是否包含非数字或无穷大)
    if (p_ros.array().isNaN().any() || p_ros.array().isInf().any()) {
        RCLCPP_ERROR(this->get_logger(), "有效性检查失败：p_ros 包含 NaN 或 Inf!");
    }

    // 4. 检查四元数 q_ros (确保已归一化，且不包含非法值)
    if (std::abs(q_ros.norm() - 1.0f) > 0.1) {
        // 如果模长偏离 1 太远，说明旋转矩阵转换出错
        RCLCPP_ERROR(this->get_logger(), "有效性检查失败：四元数未归一化 (norm: %.2f)", q_ros.norm());
    }

    // 打印当前位姿 (ROS 坐标系)
    RCLCPP_INFO(this->get_logger(), "Pos: [x:%.2f, y:%.2f, z:%.2f] | Quat: [x:%.2f, y:%.2f, z:%.2f, w:%.2f]",
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
    // 发布map -> odm tf
    this->PublishMap2OdomTF(p_ros.cast<double>(),q_ros.cast<double>(),stamp,m_bTbcLoaded);

    // --- 7. 发布 Odometry 消息 (作为 EKF 的视觉里程计输入) ---
    if (v_world) {
        RCLCPP_INFO(this->get_logger(), "SLAM Tracking OK: Publishing Odometry...");
        Eigen::Vector3f v_body_ros;
        Utility::ConvertSLALinearVelocityToROS(v_world,R_cv,v_body_ros);
        // 2. 检查数值完整性 (防止 NaN 和 Inf)
        if (v_body_ros.array().isNaN().any() || v_body_ros.array().isInf().any()) {
            RCLCPP_ERROR(this->get_logger(), "速度转换异常：检测到 NaN 或 Inf！跳过发布。");
            return; // 必须直接拦截，不能发布
        }

        // 3. 检查物理合理性 (逻辑校验)
        // 假设你的机器人最大速度不会超过 5m/s
        float speed_norm = v_body_ros.norm();
        if (speed_norm > 5.0f) {
            RCLCPP_WARN(this->get_logger(), "检测到异常高速: %.2f m/s，可能是 SLAM 跟踪发散", speed_norm);
            // 根据需求决定是否拦截，或者进行限幅处理
        }

        // 4. 打印调试信息 (Throttle 限制打印频率)
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
            "机体系速度 (ROS): [vx:%.3f, vy:%.3f, vz:%.3f] | 总速率: %.2f m/s",
            v_body_ros.x(), v_body_ros.y(), v_body_ros.z(), speed_norm);

        auto odom_msg = nav_msgs::msg::Odometry();

        // 7.1 设置 Header
        odom_msg.header.stamp = stamp; 
        // 修改点：设为 "odom"。告诉 EKF 这是相对于里程计原点的位姿
        odom_msg.header.frame_id = "odom"; 
        // 被观测的对象依然是机器人本体
        odom_msg.child_frame_id = "base_link";

        // 7.2 填充位姿数据
        // ❌ 不发布 pose（防止语义错误）
        // ❌ 明确告诉 EKF：不要 pose
        odom_msg.pose.pose.orientation.w = 1.0;
        for (int i = 0; i < 36; ++i)
            odom_msg.pose.covariance[i] = 1e6;

        odom_msg.twist.twist.linear.x = v_body_ros.x();
        odom_msg.twist.twist.linear.y = v_body_ros.y();
        odom_msg.twist.twist.linear.z = v_body_ros.z();
        // ===== 协方差 =====
        for (int i = 0; i < 36; i++)
            odom_msg.twist.covariance[i] = 0.0;

        odom_msg.twist.covariance[0]  = 0.05;
        odom_msg.twist.covariance[7]  = 0.05;
        odom_msg.twist.covariance[14] = 0.1;

        // ===============================
        // 5. 角速度（IMU 原始数据）
        // ===============================
        if (lastPoint)
        {
            Utility::PrintImuPoint("最后一个 IMU 点", lastPoint);
            // lastPoint->w 已经在 IMU / body 坐标系
            // 不需要 OpenCV → ROS 变换
            odom_msg.twist.twist.angular.x = lastPoint->w.x();
            odom_msg.twist.twist.angular.y = lastPoint->w.y();
            odom_msg.twist.twist.angular.z = lastPoint->w.z();

            odom_msg.twist.covariance[21] = 0.02;
            odom_msg.twist.covariance[28] = 0.02;
            odom_msg.twist.covariance[35] = 0.01;
        }
        else
        {
            odom_msg.twist.covariance[21] = 1e6;
            odom_msg.twist.covariance[28] = 1e6;
            odom_msg.twist.covariance[35] = 1e6;
        }
        // 发布数据
        m_odom_publisher->publish(odom_msg);

        static int odm_count = 0;
        odm_count++;
        if(odm_count % 10 == 0) {
            RCLCPP_INFO(this->get_logger(), "Successfully published %d odm.", odm_count);
        }
    }
    

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
