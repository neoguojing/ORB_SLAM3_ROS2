#include "monocular-slam-node.hpp"

#include<opencv2/core/core.hpp>

// IMU数据定义
// 字段,数值,单位,物理意义
// timestamp,1520504428069562682,ns (纳秒),采样时间戳（需除以 109 转为秒）
// w_RS_S_x/y/z,"-0.027, -0.090, 0.022",rad⋅s−1,角速度（陀螺仪数据）
// a_RS_S_x/y/z,"0.169, 0.237, 9.734",m⋅s−2,线加速度（加速度计数据）

using std::placeholders::_1;


MonocularSlamNode::MonocularSlamNode(ORB_SLAM3::System* pSLAM, bool useIMU,
    bool pubMap2Odom = false,bool pubOdom = true,bool pubMap2Base = false,bool pubDebugImage=false,bool pubCloudPoint=true)
:   Node("ORB_SLAM3_ROS2")
{
    m_SLAM = pSLAM;
    m_useIMU = useIMU;
    m_pub_map_odom = pubMap2Odom;
    m_pub_odom = pubOdom;
    m_pub_pos = pubMap2Base;
    m_pub_debug_image = pubDebugImage;
    m_pub_cloud_point = pubCloudPoint;
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

    if (m_useIMU) {
        m_imu_subscriber = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu/data_raw", 10,
            std::bind(&MonocularSlamNode::GrabImu, this, std::placeholders::_1));
    }

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

MonocularSlamNode::~MonocularSlamNode()
{
    // Stop all threads
    m_SLAM->Shutdown();

    // Save camera trajectory
    m_SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
}

void MonocularSlamNode::GrabImu(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    // 0. 基本空指针/时间检查（防御式编程）
    if (!msg) return;

    rclcpp::Time header_time(msg->header.stamp);
    int64_t current_ns = header_time.nanoseconds();
    double t_sec = static_cast<double>(current_ns) * 1e-9; // 精确转换

    // 1. 值有效性检查：包含 NaN/Inf。对加速度、角速度和四元数都做检查
    auto isFinite6 = [&](const sensor_msgs::msg::Imu::SharedPtr& m) {
        return std::isfinite(m->linear_acceleration.x) && std::isfinite(m->linear_acceleration.y) && std::isfinite(m->linear_acceleration.z)
            && std::isfinite(m->angular_velocity.x) && std::isfinite(m->angular_velocity.y) && std::isfinite(m->angular_velocity.z)
            && std::isfinite(m->orientation.x) && std::isfinite(m->orientation.y) && std::isfinite(m->orientation.z) && std::isfinite(m->orientation.w);
    };
    if (!isFinite6(msg)) {
        RCLCPP_ERROR(this->get_logger(), "IMU 数据包含无效数值 (NaN/Inf)，已跳过！");
        return;
    }

    // 可选：极端值保护（防止传感器短路产生的巨大值）
    const double accel_magnitude = std::sqrt(
        msg->linear_acceleration.x * msg->linear_acceleration.x +
        msg->linear_acceleration.y * msg->linear_acceleration.y +
        msg->linear_acceleration.z * msg->linear_acceleration.z);
    if (accel_magnitude > m_max_accel_magnitude_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "异常加速度 magnitude=%.2f > %.1f. 丢弃该条 IMU.", accel_magnitude, m_max_accel_magnitude_);
        return;
    }

    std::lock_guard<std::mutex> lock(m_mutex_imu);

    // 2. 单调性与去重：使用纳秒整数判断（精确）
    if (m_last_imu_ns_ >= 0) {
        if (current_ns <= m_last_imu_ns_) {
            // 小量抖动允许（例如 1 微秒以内）——但一般直接丢弃或调节时间
            if (current_ns == m_last_imu_ns_) {
                // 完全重复时间戳，通常丢弃
                RCLCPP_DEBUG(this->get_logger(), "收到重复时间戳 IMU（ns=%ld），丢弃。", (long)current_ns);
                return;
            } else {
                // 时间倒流（例如设备重置或 NTP 回拨）
                RCLCPP_WARN(this->get_logger(), "IMU 时间倒流或跳变（now=%ld, last=%ld），已丢弃。", (long)current_ns, (long)m_last_imu_ns_);
                return;
            }
        }
    }

    // 3. 推入缓冲（保持 t 使用 double，但来自整数转换）
    ORB_SLAM3::IMU::Point p(
        static_cast<float>(msg->linear_acceleration.x),
        static_cast<float>(msg->linear_acceleration.y),
        static_cast<float>(msg->linear_acceleration.z),
        static_cast<float>(msg->angular_velocity.x),
        static_cast<float>(msg->angular_velocity.y),
        static_cast<float>(msg->angular_velocity.z),
        t_sec
    );
    m_imu_buffer.push_back(p);
    m_last_imu_ns_ = current_ns;

    // 4. 裁剪缓冲：保留最近 m_max_buffer_seconds_；注意保留最后一个点以便重叠
    while (m_imu_buffer.size() > 1 && m_imu_buffer.front().t < (t_sec - m_max_buffer_seconds_)) {
        m_imu_buffer.pop_front();
    }

    // 5. 日志（节流）
    RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "收到 IMU 时间: %.6f, acc=[%.3f,%.3f,%.3f], gyro=[%.3f,%.3f,%.3f], buffer_size=%zu",
        t_sec,
        msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z,
        msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z,
        m_imu_buffer.size());
}

// 线性插值辅助函数
static ORB_SLAM3::IMU::Point
InterpImuPoint(const ORB_SLAM3::IMU::Point& p0,
               const ORB_SLAM3::IMU::Point& p1,
               double t)
{
    const double dt = p1.t - p0.t;
    if (dt <= 0.0) {
        // 时间异常，返回前一个点（但时间对齐到 t）
        return ORB_SLAM3::IMU::Point(
            cv::Point3f(p0.a.x(), p0.a.y(), p0.a.z()),
            cv::Point3f(p0.w.x(), p0.w.y(), p0.w.z()),
            t
        );
    }

    const double alpha = (t - p0.t) / dt;

    Eigen::Vector3f acc =
        static_cast<float>(1.0 - alpha) * p0.a +
        static_cast<float>(alpha)       * p1.a;

    Eigen::Vector3f gyro =
        static_cast<float>(1.0 - alpha) * p0.w +
        static_cast<float>(alpha)       * p1.w;

    return ORB_SLAM3::IMU::Point(
        cv::Point3f(acc.x(),  acc.y(),  acc.z()),
        cv::Point3f(gyro.x(), gyro.y(), gyro.z()),
        t
    );
}


/**
 * @brief 提取并对齐 IMU 数据
 * @return 过滤后的 IMU 测量矢量。如果数据未就绪则返回空。
 */
std::vector<ORB_SLAM3::IMU::Point> MonocularSlamNode::SyncImuData(double t_image)
{
    std::vector<ORB_SLAM3::IMU::Point> vFilteredImu;
    std::lock_guard<std::mutex> lock(m_mutex_imu);

    if (m_imu_buffer.empty())
        return {};

    // 1. IMU 是否滞后于图像
    if (m_imu_buffer.back().t < t_image - 0.01) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 500,
            "IMU 数据滞后! 最新 IMU: %.6f, 图像: %.6f, Δ=%.3f s",
            m_imu_buffer.back().t, t_image,
            t_image - m_imu_buffer.back().t);
        return {};
    }

    // 2. 找到第一个 t > t_image 的 IMU
    auto it = std::upper_bound(
        m_imu_buffer.begin(), m_imu_buffer.end(),
        t_image,
        [](double t, const ORB_SLAM3::IMU::Point& p) {
            return t < p.t;
        });

    // 极端情况：首个 IMU 就晚于图像
    if (it == m_imu_buffer.begin()) {
        RCLCPP_WARN(
            this->get_logger(),
            "SyncImuData: first IMU t=%.6f > image t=%.6f",
            m_imu_buffer.front().t, t_image);
        return {};
    }

    // 3. 收集 [begin, it) 的真实 IMU
    double last_t = -1.0;
    for (auto iter = m_imu_buffer.begin(); iter != it; ++iter) {
        if (last_t > 0) {
            double dt = iter->t - last_t;
            if (dt > m_max_imu_dt_) {
                RCLCPP_WARN(
                    this->get_logger(),
                    "IMU time gap detected: dt=%.4f s (>%.4f)",
                    dt, m_max_imu_dt_);
            }
        }
        vFilteredImu.push_back(*iter);
        last_t = iter->t;
    }

    // 4. 插值生成 t_image 对齐点（不覆盖真实 IMU）
    auto prev_it = std::prev(it);
    if (it != m_imu_buffer.end()) {
        ORB_SLAM3::IMU::Point interp =
            InterpImuPoint(*prev_it, *it, t_image);
        vFilteredImu.push_back(interp);
    }

    // 5. 裁剪 buffer：保留 prev_it 作为重叠点
    m_imu_buffer.erase(m_imu_buffer.begin(), prev_it);

    RCLCPP_DEBUG_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "SyncImuData: return %zu IMU, buffer left %zu",
        vFilteredImu.size(), m_imu_buffer.size());

    return vFilteredImu;
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

/**
 * @brief 执行 SLAM 跟踪
 * @return 得到的相机位姿 Tcw
 */
Sophus::SE3f MonocularSlamNode::ExecuteTracking(const cv::Mat& im, double t_image, const std::vector<ORB_SLAM3::IMU::Point>& vImu) {
    Sophus::SE3f Tcw;
    auto t1 = std::chrono::high_resolution_clock::now();

    if (!m_useIMU) {
        Tcw = m_SLAM->TrackMonocular(im, t_image);
    } else {
        Tcw = m_SLAM->TrackMonocular(im, t_image, vImu);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    m_last_elapsed = std::chrono::duration<double>(t2 - t1).count();

    return Tcw;
}

/**
 * @brief 提取当前帧的运动状态（速度等）
 */
bool MonocularSlamNode::ExtractMotionInfo(Eigen::Vector3f& v_world) {
    ORB_SLAM3::Tracking* pTracker = m_SLAM->GetTracker();
    if (pTracker && m_useIMU) {
        v_world = pTracker->mCurrentFrame.GetVelocity();
        // 阈值检查：超过 20/s 的速度通常是初始化瞬间的数值爆炸
        if (v_world.array().isFinite().all() && v_world.norm() < 20.0f) {
            return true;
        }
    }
    return false;
}


void MonocularSlamNode::ProcessImage(const cv::Mat& im, const rclcpp::Time& stamp)
{
    double t_image = Utility::StampToSec(stamp);
    // 如果这条不打印，说明是之前的 QoS 不匹配或网络包过大丢失问题
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
        "收到图像消息! 时间戳: %.3f, 宽度: %d, 高度: %d", 
        t_image, im.cols, im.rows);

    // 尝试获取Tbc
    if (!m_bTbcLoaded && m_SLAM->GetSetting()) {
        m_Tbc = m_SLAM->GetSetting()->Tbc();
        m_bTbcLoaded = true;
        Utility::PrintSophusSE3("Tbc 外参", m_Tbc);
    }

    
    // 1. 同步 IMU
    std::vector<ORB_SLAM3::IMU::Point> vImu = m_useIMU ? this->SyncImuData(t_image) : std::vector<ORB_SLAM3::IMU::Point>();
    if (m_useIMU && vImu.size() < 2) {
        RCLCPP_WARN(this->get_logger(), "IMU 数据不足 (%ld 个)，跳过该帧", vImu.size());
        return;
    }

    // 2. 跟踪
    Sophus::SE3f Tcw = ExecuteTracking(im, t_image, vImu);
    
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
    RCLCPP_INFO_THROTTLE(this->get_logger(),*this->get_clock(), 3000, "当前帧提取到的特征点数: %d", nFeatures);
    
    // 4. 检查状态并发布
    int state = m_SLAM->GetTrackingState();
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
        "SLAM 跟踪状态: %d (9=OK), 耗时: %.4f 秒", state, m_last_elapsed);
    
    double now = this->now().seconds();
    if (state == ORB_SLAM3::Tracking::OK) {
        m_lost_start_time = -1;
        
        Eigen::Vector3f v_world;
        bool has_vel = this->ExtractMotionInfo(v_world);
        const ORB_SLAM3::IMU::Point* imu_ptr = vImu.empty() ? nullptr : &vImu.back();

        // 核心数据分发 (之前讨论的 map->odom 发布就在这里面)
        this->HandleSlamOutput(Tcw, stamp, has_vel ? &v_world : nullptr, imu_ptr);
        if (m_pub_cloud_point)
            this->PublishMapPoints(stamp);
        // 每隔几帧发布一次点云，减轻树莓派压力
        if (m_frame_count++ % 5 == 0) {
            if (m_pub_debug_image)
                this->PublishImageData(stamp);
        }
    } else if (state == ORB_SLAM3::Tracking::LOST) {
        RCLCPP_WARN(this->get_logger(), "SLAM 跟踪丢失: %d！", state);
        if (m_lost_start_time < 0)
            m_lost_start_time = now;
        
        if (now - m_lost_start_time > 2.0) {
            m_SLAM->ResetActiveMap();
            m_lost_start_time = -1;
        }

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


void MonocularSlamNode::PublishMapPoints(const rclcpp::Time& stamp)
{
    // 1. 获取所有地图点
    std::vector<ORB_SLAM3::MapPoint*> vpMapPoints = m_SLAM->GetTrackedMapPoints();
    if (vpMapPoints.empty()) return;

    auto cloud_msg = sensor_msgs::msg::PointCloud2();
    cloud_msg.header.stamp = stamp;
    cloud_msg.header.frame_id = "map"; // 地图点是在 map 坐标系下的

    sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
    modifier.setPointCloud2FieldsByString(1, "xyz");

    // 预过滤有效点
    std::vector<Eigen::Vector3f> valid_points;
    valid_points.reserve(vpMapPoints.size());
    for (auto pMP : vpMapPoints) {
        if (pMP && !pMP->isBad()) {
            // 使用与 Pose 统一的转换矩阵
            // 可选：高度过滤（假设 z 是高度）
            // if (pos_ros.z() < 0.05 || pos_ros.z() > 2.0) continue;
            Eigen::Vector3f pos = m_R_vis_ros * pMP->GetWorldPos();
            if (pos.norm() > 50.0f) continue;
            valid_points.push_back(pos);
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

/**
 * @brief 从 TF 树中查询静态外参 (base_link -> sensor)
 * @throws std::runtime_error 如果外参缺失，强制抛出异常，防止静默失败
 */
Sophus::SE3f MonocularSlamNode::GetStaticTransformAsSophus(const std::string& target_frame) {

    if (static_tf_cache_.count(target_frame)) {
        return static_tf_cache_.at(target_frame);
    }

    try {
        // 使用 TimePointZero 获取最新静态变换，不引入时间戳抖动
        auto msg = tf_buffer_->lookupTransform("base_link", target_frame, tf2::TimePointZero);
        
        // 注意：tf2 旋转分量顺序是 x, y, z, w
        Eigen::Quaternionf q(
            msg.transform.rotation.w,
            msg.transform.rotation.x,
            msg.transform.rotation.y,
            msg.transform.rotation.z);
            
        Eigen::Vector3f t(
            msg.transform.translation.x,
            msg.transform.translation.y,
            msg.transform.translation.z);

        // 2. 打印获取到的关系 (只在第一次获取时打印)
        RCLCPP_INFO(this->get_logger(), "Successfully cached static TF [base_link -> %s]:", target_frame.c_str());
        RCLCPP_INFO(this->get_logger(), " - Translation: [%.3f, %.3f, %.3f]", t.x(), t.y(), t.z());
        RCLCPP_INFO(this->get_logger(), " - Rotation (Euler RPY): [%.3f, %.3f, %.3f]", 
                    q.toRotationMatrix().eulerAngles(0, 1, 2).cast<float>().x(),
                    q.toRotationMatrix().eulerAngles(0, 1, 2).cast<float>().y(),
                    q.toRotationMatrix().eulerAngles(0, 1, 2).cast<float>().z());
        Sophus::SE3f T_base_target(q, t);
        // 3. 存入缓存
        static_tf_cache_[target_frame] = T_base_target;
        return T_base_target;
    } catch (const tf2::TransformException& ex) {
        // 1. 记录 FATAL 级别日志（比 ERROR 更严重）
        RCLCPP_FATAL(
            this->get_logger(), 
            "CRITICAL ERROR: Static transform 'base_link' -> '%s' is MISSING! "
            "SLAM pose alignment is impossible without this. Error: %s", 
            target_frame.c_str(), ex.what());
        
        // 2. 强制抛出异常，不再返回单位阵
        throw std::runtime_error("Missing essential static TF: base_link to " + target_frame);
    }
}

void MonocularSlamNode::HandleSlamOutput(const Sophus::SE3f& Tcw, const rclcpp::Time& stamp,
    const Eigen::Vector3f* v_world,const ORB_SLAM3::IMU::Point* lastPoint) {
    Eigen::Vector3f p_base_ros;
    Eigen::Quaternionf q_base_ros;
    Eigen::Matrix3f R_cv;

    try {
        // 根据模式获取对应的外参，缺失会抛出异常跳转到 catch
        if (m_bTbcLoaded) {
            Sophus::SE3f T_base_imu = this->GetStaticTransformAsSophus("imu_link");
            Utility::ConvertSLAMPoseToBaseLink(Tcw, p_base_ros, q_base_ros,R_cv, &m_Tbc, &T_base_imu, nullptr);
        } else {
            Sophus::SE3f T_base_cam = this->GetStaticTransformAsSophus("camera_link_optical");
            Utility::ConvertSLAMPoseToBaseLink(Tcw, p_base_ros, q_base_ros,R_cv, nullptr, nullptr, &T_base_cam);
        }

        // --- 修复点：双重归一化保险 ---
        // 虽然 Convert 内部已经 normalize，但在发布前再次保险，彻底杜绝 float 累积漂移导致的 TF 失真
        q_base_ros.normalize();

        // 3. 检查平移向量 p_ros (是否包含非数字或无穷大)
        if (p_base_ros.array().isNaN().any() || p_base_ros.array().isInf().any()) {
            RCLCPP_ERROR(this->get_logger(), "有效性检查失败：p_ros 包含 NaN 或 Inf!");
        }

        // 4. 检查四元数 q_ros (确保已归一化，且不包含非法值)
        if (std::abs(q_base_ros.norm() - 1.0f) > 0.1) {
            // 如果模长偏离 1 太远，说明旋转矩阵转换出错
            RCLCPP_ERROR(this->get_logger(), "有效性检查失败：四元数未归一化 (norm: %.2f)", q_base_ros.norm());
        }

        // 打印当前位姿 (ROS 坐标系)
         RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Pos: [x:%.2f, y:%.2f, z:%.2f] | Quat: [x:%.2f, y:%.2f, z:%.2f, w:%.2f]",
                    p_base_ros.x(), p_base_ros.y(), p_base_ros.z(), q_base_ros.x(), q_base_ros.y(), q_base_ros.z(), q_base_ros.w());

        // 发布 map -> odm
        if (m_pub_map_odom)
            this->PublishMap2OdomTF(p_base_ros.cast<double>(), q_base_ros.cast<double>(), stamp);
        // 发布 odm
        if (m_pub_odom)
            this->PublishOdm(R_cv,v_world,lastPoint, stamp);
        // 发布 pos
        if (m_pub_pos)
            this->PublishPos(p_base_ros.cast<double>(), q_base_ros.cast<double>(), stamp);
        


    } catch (const std::exception& e) {
        // 捕获 GetStaticTransformAsSophus 抛出的异常，防止节点崩溃，但通过日志警告
        RCLCPP_ERROR_THROTTLE(
            this->get_logger(), *this->get_clock(), 5000, 
            "Slam Pose Processing Aborted: %s", e.what());
    }
}

/**
 * @brief 发布 map -> odom 变换，消除里程计漂移
 * @param p_map_base SLAM计算出的 base_link 在 map 系下的平移
 * @param q_map_base SLAM计算出的 base_link 在 map 系下的旋转
 * @param stamp      当前图像帧的时间戳
 */
/**
 * @brief 发布 map -> odom 变换，用于消除里程计漂移（Nav2 / EKF 友好版）
 * @param p_map_base  SLAM 计算得到的 base_link 在 map 坐标系下的位置
 * @param q_map_base  SLAM 计算得到的 base_link 在 map 坐标系下的姿态
 * @param stamp       图像帧对应的时间戳（必须与 SLAM 位姿同步）
 */
void MonocularSlamNode::PublishMap2OdomTF(
    const Eigen::Vector3d& p_map_base,
    const Eigen::Quaterniond& q_map_base,
    const rclcpp::Time& stamp)
{
    // ===============================
    // 1. SLAM: Map -> Base（压平成 2D）
    // ===============================

    // 从 SLAM 四元数中只提取 yaw
    Eigen::Matrix3d R_map_base = q_map_base.toRotationMatrix();
    double slam_yaw = std::atan2(R_map_base(1, 0), R_map_base(0, 0));

    tf2::Transform map_to_base_2d;
    map_to_base_2d.setOrigin(tf2::Vector3(
        p_map_base.x(),
        p_map_base.y(),
        0.0                    // ⭐ 强制 z = 0
    ));

    tf2::Quaternion q_slam_yaw;
    q_slam_yaw.setRPY(0.0, 0.0, slam_yaw);  // ⭐ 只保留 yaw
    q_slam_yaw.normalize();
    map_to_base_2d.setRotation(q_slam_yaw);

    // ===============================
    // 2. EKF: 查询 Odom -> Base_footprint
    // ===============================
    geometry_msgs::msg::TransformStamped odom_to_base_msg;
    try {
        odom_to_base_msg = tf_buffer_->lookupTransform(
            "odom",
            "base_footprint",   // ⭐ 强烈推荐使用 base_footprint
            stamp,
            rclcpp::Duration(0, 100 * 1000 * 1000) // 100 ms
        );
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "Failed to lookup odom->base_footprint, skip map->odom: %s",
            ex.what()
        );
        return;
    }

    tf2::Transform odom_to_base;
    tf2::fromMsg(odom_to_base_msg.transform, odom_to_base);

    // ===============================
    // 2.1 EKF 位姿防御性压平（关键）
    // ===============================
    double o_roll, o_pitch, o_yaw;
    tf2::Matrix3x3(odom_to_base.getRotation()).getRPY(
        o_roll, o_pitch, o_yaw
    );

    tf2::Transform odom_to_base_2d;
    odom_to_base_2d.setOrigin(tf2::Vector3(
        odom_to_base.getOrigin().x(),
        odom_to_base.getOrigin().y(),
        0.0                    // ⭐ 再次强制 z = 0
    ));

    tf2::Quaternion q_odom_yaw;
    q_odom_yaw.setRPY(0.0, 0.0, o_yaw);      // ⭐ 只保留 yaw
    q_odom_yaw.normalize();
    odom_to_base_2d.setRotation(q_odom_yaw);

    // ===============================
    // 3. 反算 Map -> Odom（2D ⨉ 2D）
    //
    // 数学关系：
    //   T_map_base = T_map_odom * T_odom_base
    //   => T_map_odom = T_map_base * T_odom_base⁻¹
    // ===============================
    tf2::Transform map_to_odom =
        map_to_base_2d * odom_to_base_2d.inverse();

    // ===============================
    // 4. 发布 TF
    // ===============================
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = "map";
    tf_msg.child_frame_id = "odom";
    tf_msg.transform = tf2::toMsg(map_to_odom);

    tf_broadcaster_->sendTransform(tf_msg);

    // ===============================
    // 5. 调试日志（节流）
    // ===============================
    RCLCPP_DEBUG_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Published map->odom (2D): "
        "x=%.2f y=%.2f yaw=%.2f deg",
        map_to_odom.getOrigin().x(),
        map_to_odom.getOrigin().y(),
        o_yaw * 180.0 / M_PI
    );
}


void MonocularSlamNode::PublishOdm(
    const Eigen::Matrix3f& R_cv,
    const Eigen::Vector3f* v_world,
    const ORB_SLAM3::IMU::Point* lastPoint,
    const rclcpp::Time& stamp)
{

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
        odom_msg.header.frame_id = "base_link"; 
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
}

// 等价于map->base_link
void MonocularSlamNode::PublishPos(
    const Eigen::Vector3d& p_map_base,
    const Eigen::Quaterniond& q_map_base,
    const rclcpp::Time& stamp) 
{
    // 5. 填充并发布 PoseStamped 消息(调试使用)
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = "map"; // 全局地图坐标系
    
    pose_msg.pose.position.x = p_map_base.x();
    pose_msg.pose.position.y = p_map_base.y();
    pose_msg.pose.position.z = p_map_base.z();
    
    pose_msg.pose.orientation.x = q_map_base.x();
    pose_msg.pose.orientation.y = q_map_base.y();
    pose_msg.pose.orientation.z = q_map_base.z();
    pose_msg.pose.orientation.w = q_map_base.w();

    m_pose_publisher->publish(pose_msg);
}