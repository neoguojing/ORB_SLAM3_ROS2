#ifndef __UTILITY_HPP__
#define __UTILITY_HPP__

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include "rclcpp/rclcpp.hpp"

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
      RCLCPP_INFO(logger, "Step 1: Start conversion");
      Sophus::SE3f Twc = Tcw.inverse();
      RCLCPP_INFO(logger, "Step 2: Inverse done");
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
  }

};

#endif
