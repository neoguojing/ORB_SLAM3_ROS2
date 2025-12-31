#ifndef __UTILITY_HPP__
#define __UTILITY_HPP__

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include "rclcpp/rclcpp.hpp"
#include "ImuTypes.h"       // 必须包含，定义了 IMU::Point

class Utility
{
public:
  static double StampToSec(builtin_interfaces::msg::Time stamp)
  {
    // double seconds = stamp.sec + (stamp.nanosec * pow(10,-9));
    // return seconds;
    return rclcpp::Time(stamp).seconds();
  }

    /**
     * @brief 将 ORB-SLAM3 位姿转换为 ROS base_link 格式
     * @param Tcw         [输入] SLAM输出的相机位姿 (World -> Camera)
     * @param p_ros       [输出] map -> base_link 的平移 (ROS轴)
     * @param q_ros       [输出] map -> base_link 的旋转 (ROS轴)
     * @param Tbc         [可选] IMU模式外参 (Body -> Camera, OpenCV轴)
     * @param T_base_imu  [可选] 机器底座到IMU的安装关系 (base_link -> imu_link, ROS轴)
     * @param T_base_cam  [可选] 机器底座到相机的安装关系 (base_link -> camera_optical, 混合轴)
     * 
     * Tcw	Camera → World	OpenCV 相机轴
     * Twc	World → Camera	OpenCV 相机轴
     * Tbc	Body(IMU) → Camera	OpenCV 相机轴
     * Twb	World → Body(IMU)	OpenCV 相机轴
     * R_cv	World→输出frame 的旋转，但仍是 OpenCV 轴	OpenCV
     */
    static void ConvertSLAMPoseToBaseLink(
        const Sophus::SE3f& Tcw,
        Eigen::Vector3f& p_ros,
        Eigen::Quaternionf& q_ros,
        Eigen::Matrix3f& R_cv,
        const Sophus::SE3f* Tbc = nullptr,
        const Sophus::SE3f* T_base_imu = nullptr,
        const Sophus::SE3f* T_base_cam = nullptr) 
    {
        // 1. 定义 OpenCV -> ROS 的轴向转换矩阵 (用于将光学坐标系下的结果转到 ROS 语义下)
        static const Eigen::Matrix3f m_R_vis_ros = (Eigen::Matrix3f() << 0, 0, 1, -1, 0, 0, 0, -1, 0).finished();
        
        // 2. 得到相机在世界系下的位姿 Twc (World -> Camera, OpenCV 轴向)
        Sophus::SE3f Twc = Tcw.inverse();
        R_cv = Twc.rotationMatrix()
        if (Tbc && T_base_imu) {
            // --- 情况 A: IMU 模式 ---
            // 先算 Map -> IMU (ROS轴)，再算 Map -> Base (ROS轴)
            Sophus::SE3f Twb_cv = Twc * (Tbc->inverse());
            
            Eigen::Vector3f p_imu_ros = m_R_vis_ros * Twb_cv.translation();
            Eigen::Matrix3f R_imu_ros = m_R_vis_ros * Twb_cv.rotationMatrix() * m_R_vis_ros.transpose();
            Sophus::SE3f T_map_imu_ros(R_imu_ros, p_imu_ros);

            Sophus::SE3f T_map_base = T_map_imu_ros * (T_base_imu->inverse());
            p_ros = T_map_base.translation();
            q_ros = Eigen::Quaternionf(T_map_base.rotationMatrix());
        } 
        else if (T_base_cam) {
            // --- 情况 B: 纯视觉模式 (补全部分) ---
            // 逻辑：T_map_base = T_map_camera_optical * T_camera_optical_to_base
            // 注意：Twc 是 OpenCV 轴向，T_base_cam 通常也是定义为从 ROS Base 到 CV Cam
            
            // 1. 在 OpenCV 空间计算 base_link 的位置
            // T_map_base_cv = Twc * T_base_cam^-1
            Sophus::SE3f T_map_base_cv = Twc * (T_base_cam->inverse());

            // 2. 转换到 ROS 轴向空间
            // 这一步非常关键：它将 SLAM 定义的“前进方向”对齐到 ROS 定义的“前进方向”
            p_ros = m_R_vis_ros * T_map_base_cv.translation();
            Eigen::Matrix3f R_base_ros = m_R_vis_ros * T_map_base_cv.rotationMatrix() * m_R_vis_ros.transpose();
            
            q_ros = Eigen::Quaternionf(R_base_ros);
        } 
        else {
            // --- 情况 C: 无外参，退化输出 ---
            p_ros = Twc.translation();
            q_ros = Eigen::Quaternionf(Twc.rotationMatrix());
        }

        q_ros.normalize();
    }

  /**
   * @brief 将slam系线速度转换为 ROS 坐标系
   *
   * 参数：
   * @param v_slam_world  slam坐标系下线速度指针 (可为 nullptr),相机光心在 SLAM 世界中的瞬时线速度
   * @param R_cv     相机坐标系在 SLAM 世界坐标系中的朝向
   * @param v_ros    输出：ROS(base_link) 坐标系线速度
   *
   * 数学公式：
   *   v_temp = R_cvᵀ * v_slam_world
   *   v_ros = R_vis_ros * v_temp
   */
  static void ConvertSLALinearVelocityToROS(
      const Eigen::Vector3f* v_slam_world,
      const Eigen::Matrix3f& R_cv,
      Eigen::Vector3f& v_ros)
  { 
      auto logger = rclcpp::get_logger("utility_velocity");
      // 1. 拦截空指针和无效输入 (预防 NaN 传播)
      if (!v_slam_world || !v_slam_world->allFinite() || !R_cv.allFinite()) {
          // 如果输入数据不对，直接返回全 0 速度，不要进行矩阵乘法
          RCLCPP_WARN(logger, "v_slam_world 或 R_cv 非法，返回零速度");
          return; 
      }

      // 2. 检查 R_cv 的合法性 (虽然它是 Eigen 矩阵，但如果它不满足旋转矩阵特性，也会出问题)
      // 如果行列式为 0，说明矩阵不可逆或已损坏
      if (std::abs(R_cv.determinant()) < 1e-6) {
          RCLCPP_WARN(logger, "R_cv 非法（行列式接近零），返回零速度");
          return;
      }
      // OPENCV-> ROS 坐标轴基变换矩阵
      Eigen::Matrix3f m_R_vis_ros;
      m_R_vis_ros << 0, 0, 1,
                    -1, 0, 0,
                    0,-1, 0;

      v_ros.setZero();

      if (v_slam_world) {
          // R_cv.transpose()：SLAM 世界中的向量，在相机自己坐标系下怎么表示 camera link
          // v_temp：相机自己怎么感觉这股运动
          // v_slam_world： slam怎么看你的速度
          Eigen::Vector3f v_temp = R_cv.transpose() * (*v_slam_world); // slam系 -> camera
          // 转换为ROS坐标轴，camera如何感觉自己的运动
          v_ros = m_R_vis_ros * v_temp;                           // OPENCV -> ROS
      }

      RCLCPP_INFO(logger, "ConvertSLALinearVelocityToROS finished");
  }

  // --- 打印 Eigen::Matrix3f (旋转矩阵) ---
  static void PrintEigenMatrix3f(const std::string& name, const Eigen::Matrix3f& mat) {
      auto logger = rclcpp::get_logger("utility_debug");
      RCLCPP_INFO(logger, "--- %s ---", name.c_str());
      for (int i = 0; i < 3; ++i) {
          RCLCPP_INFO(logger, "  [%.4f, %.4f, %.4f]", 
                      mat(i, 0), mat(i, 1), mat(i, 2));
      }
  }

  // --- 打印 Sophus::SE3f (位姿矩阵) ---
  static void PrintSophusSE3(const std::string& name, const Sophus::SE3f& T) {
      auto logger = rclcpp::get_logger("utility_debug");
      Eigen::Vector3f t = T.translation();
      Eigen::Matrix3f R = T.rotationMatrix();
      // 转换为欧拉角 (Z-Y-X 顺序: yaw, pitch, roll) 方便人类理解
      Eigen::Vector3f euler = R.eulerAngles(2, 1, 0) * 180.0 / M_PI;

      RCLCPP_INFO(logger, "==== %s ====", name.c_str());
      RCLCPP_INFO(logger, "平移 (x, y, z): [%.4f, %.4f, %.4f]", t.x(), t.y(), t.z());
      RCLCPP_INFO(logger, "旋转 (yaw, pitch, roll)°: [%.2f, %.2f, %.2f]", euler[0], euler[1], euler[2]);
      
      // 打印原始 3x3 旋转矩阵
      for (int i = 0; i < 3; ++i) {
          RCLCPP_INFO(logger, "R[%d]: [%.4f, %.4f, %.4f]", 
                      i, R(i, 0), R(i, 1), R(i, 2));
      }
  }

  /**
 * @brief 打印 Eigen::Vector3f (专门用于速度、位置向量)
 */
  static void PrintVector3f(const std::string& name, const Eigen::Vector3f& vec) {
      RCLCPP_INFO(rclcpp::get_logger("utility_debug"), "[DEBUG] %s: [X: %.4f, Y: %.4f, Z: %.4f] | 模长: %.4f", 
                  name.c_str(), vec.x(), vec.y(), vec.z(), vec.norm());
  }

  /**
   * @brief 打印 ORB_SLAM3::IMU::Point 数据
   * @param label 打印标签，用于区分不同的调用位置
   * @param p IMU 点指针
   */
  static void PrintImuPoint(const std::string& label, const ORB_SLAM3::IMU::Point* p) {
      auto logger = rclcpp::get_logger("utility_debug");
      if (!p) {
          RCLCPP_WARN(logger, "[IMU DEBUG] %s: Point is NULL!", label.c_str());
          return;
      }

      // 提取数据
      double ts = p->t;                      // 时间戳
      Eigen::Vector3f acc = p->a;            // 线加速度
      Eigen::Vector3f gyr = p->w;            // 角速度

      RCLCPP_INFO(logger, "==== [IMU Point: %s] ====", label.c_str());
      RCLCPP_INFO(logger, "时间戳 (t): %.6f", ts);
      
      // 打印线加速度 (Acceleration)
      // 正常静止状态下，应该能看到一个轴接近 9.8 (重力)
      RCLCPP_INFO(logger, "线加速度 a (x,y,z): [%.4f, %.4f, %.4f] m/s^2", 
                  acc.x(), acc.y(), acc.z());
      
      // 打印角速度 (Gyroscope)
      // 机器人静止时，这些值应接近 0
      RCLCPP_INFO(logger, "角速度 w (x,y,z): [%.4f, %.4f, %.4f] rad/s", 
                  gyr.x(), gyr.y(), gyr.z());
      
      RCLCPP_INFO(logger, "-----------------------------------------");
  }

};

#endif
