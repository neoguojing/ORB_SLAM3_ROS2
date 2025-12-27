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
   * @brief 将 ORB-SLAM3 输出的OPENCV位姿转换为 ROS 坐标系位姿，参考系依然是SLAM
   *
   * 功能：
   *   - 处理 SLAM 输出的 camera 坐标系位姿 Tcw (slam_world->camera) 它描述的是从 世界坐标系 (World/Map) 到 当前相机坐标系 (Camera) 的变换。
   *   - 可选考虑 Camera->Body(IMU) 外参 Tbc
   *   - 输出：SLAM坐标系下的平移向量和旋转四元数，使用ROS轴方向
   *
   * 参数说明：
   * @param Tcw        ORB-SLAM3 输出位姿，SE3f，camera_link，相机观察的运动姿态
   * @param m_R_vis_ros 固定旋转矩阵，用于 OPENCV → ROS 坐标轴转换
   * @param R_cv       输出：SLAM 坐标系下的相机光心怎么旋转，但用的是 OpenCV 轴方向约定
   * @param p_ros      输出：SLAM 坐标系下的平移向量 (x,y,z)，使用ROS轴方向, map，slam观察的运动姿态
   * @param q_ros      输出：SLAM 坐标系下的旋转四元数，使用ROS轴方向 map，slam观察的运动姿态
   * @param Tbc        可选 Camera->Body(SE3f) 外参，如果没有可传 nullptr
   *
   * 使用说明：
   *   - 没有 Tbc：适合 base_link 就在 camera 原点的情况
   *   - 有 Tbc：适合 IMU/机器人本体坐标系与相机不重合的情况
   */
  static void ConvertSLAMPoseToROS(
      const Sophus::SE3f& Tcw,
      Eigen::Matrix3f& R_cv,
      Eigen::Vector3f& p_ros,
      Eigen::Quaternionf& q_ros,
      const Sophus::SE3f* Tbc = nullptr)
  {
      auto logger = rclcpp::get_logger("utility_pose");
      // OPENCV-> ROS 坐标轴基变换矩阵
      Eigen::Matrix3f m_R_vis_ros;
      m_R_vis_ros << 0, 0, 1,   // ROS X = CV Z
                    -1, 0, 0,   // ROS Y = -CV X
                    0,-1, 0;   // ROS Z = -CV Y
      // 1. 扭转刚体变换
      // Tcw: slam_world -> camera   参考系是camera
      // Twc: camera -> slam_world   参考系是slam_world
      Sophus::SE3f Twc = Tcw.inverse();
      // 3. 如果存在 Camera->Body 外参，将位姿从相机系转换到机器人本体/IMU系
      // 将slam下的相机位置转换为body的位置
      Sophus::SE3f Twb;
      if (Tbc) {
        // 使用 std::stringstream 格式化矩阵
        std::stringstream ss;
        ss << "\n" << Tbc->matrix(); // 获取 4x4 矩阵
        RCLCPP_INFO(logger, "--- 当前加载的外参 Tbc (Body to Camera) ---%s", ss.str().c_str());

        Twb = Twc * (*Tbc);            // world -> body
        RCLCPP_INFO(logger, "Step 3: Tbc multiply done");
      } else {
          Twb = Twc;                     // 没有外参就当 camera == body
      }
      // 2. 提取slam_world 坐标系下的平移和旋转
      Eigen::Vector3f p_cv = Twb.translation();     // 可以理解为slam坐标系下相机光心的“位移向量” 但用的是 OpenCV 轴方向约定
      R_cv = Twb.rotationMatrix();  // 相机光心在slam坐标系中怎么旋转，但用的是 OpenCV 轴方向约定
      // 4. 将 OPENCV的坐标轴换为ROS坐标轴
      // p_ros = R_vis_ros * p_cv
      p_ros = m_R_vis_ros * p_cv;
      // R_ros = R_vis_ros * R_cv * R_vis_ros^T
      Eigen::Matrix3f R_ros = m_R_vis_ros * R_cv * m_R_vis_ros.transpose();
      // 5. 转换为 ROS 四元数
      q_ros = Eigen::Quaternionf(R_ros);
      q_ros.normalize();  // 防止数值误差导致非单位四元数
      RCLCPP_INFO(logger, "ConvertSLAMPoseToROS finished");
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
