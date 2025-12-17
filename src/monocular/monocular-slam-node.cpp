#include "monocular-slam-node.hpp"

#include<opencv2/core/core.hpp>

using std::placeholders::_1;

MonocularSlamNode::MonocularSlamNode(ORB_SLAM3::System* pSLAM)
:   Node("ORB_SLAM3_ROS2")
{
    m_SLAM = pSLAM;
    // std::cout << "slam changed" << std::endl;
    m_image_subscriber = this->create_subscription<ImageMsg>(
        "camera",
        10,
        std::bind(&MonocularSlamNode::GrabImage, this, std::placeholders::_1));
    
    // 新增：初始化发布者
    m_pose_publisher = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "slam_pose", // Nav2 通常期望定位信息
        10
    );
    std::cout << "slam changed" << std::endl;
}

MonocularSlamNode::~MonocularSlamNode()
{
    // Stop all threads
    m_SLAM->Shutdown();

    // Save camera trajectory
    m_SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
}

void MonocularSlamNode::GrabImage(const ImageMsg::SharedPtr msg)
{
    // Copy the ros image message to cv::Mat.
    try
    {
        m_cvImPtr = cv_bridge::toCvCopy(msg);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    std::cout<<"one frame has been sent"<<std::endl;
    
    // 1. 获取位姿 (返回的是 Sophus::SE3f)
    Sophus::SE3f Tcw = m_SLAM->TrackMonocular(m_cvImPtr->image, Utility::StampToSec(msg->header.stamp));

    // 2. 检查跟踪状态
    // 注意：Tcw 本身不代表状态，需调用 GetTrackingState()
    if (m_SLAM->GetTrackingState() == ORB_SLAM3::Tracking::OK)
    {
        // 3. 获取逆变换 (相机到世界 -> 世界到相机)
        // Twc 表示相机在世界坐标系中的位姿
        Sophus::SE3f Twc = Tcw.inverse();

        // 4. 填充 ROS 2 消息
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header = msg->header;
        pose_msg.header.frame_id = "map";

        // 提取平移 (Translation)
        Eigen::Vector3f trans = Twc.translation();
        pose_msg.pose.position.x = trans.x();
        pose_msg.pose.position.y = trans.y();
        pose_msg.pose.position.z = trans.z();

        // 提取旋转并转换为四元数 (Quaternion)
        Eigen::Quaternionf q(Twc.unit_quaternion());
        pose_msg.pose.orientation.x = q.x();
        pose_msg.pose.orientation.y = q.y();
        pose_msg.pose.orientation.z = q.z();
        pose_msg.pose.orientation.w = q.w();

        // 5. 发布消息
        m_pose_publisher->publish(pose_msg);
        std::cout << "Published new SLAM pose: "
          << "Time: " << pose_msg.header.stamp.sec << "." << pose_msg.header.stamp.nanosec
          << " | Pos: [" << pose_msg.pose.position.x << ", " 
                         << pose_msg.pose.position.y << ", " 
                         << pose_msg.pose.position.z << "]"
          << " | Quat: [" << pose_msg.pose.orientation.x << ", "
                          << pose_msg.pose.orientation.y << ", "
                          << pose_msg.pose.orientation.z << ", "
                          << pose_msg.pose.orientation.w << "]"
          << std::endl;
    }
}
