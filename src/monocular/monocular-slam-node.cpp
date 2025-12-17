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
    
    cv::Mat Tcw = m_SLAM->TrackMonocular(m_cvImPtr->image, Utility::StampToSec(msg->header.stamp));
    // 只有在跟踪成功时才发布位姿
    if (!Tcw.empty())
    {
        // 1. 从 SLAM 系统获取当前相机位姿 (Tcw)
        cv::Mat Tcw = m_SLAM->GetLastFrame().GetPose();
        
        // ORB-SLAM3 返回的是 T_cw (相机到世界)，但 ROS 需要 T_wc (世界到相机)，所以需要求逆
        cv::Mat Twc = Tcw.inv();

        // 2. 将 cv::Mat 转换为 ROS 2 位姿消息
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header = msg->header;
        pose_msg.header.frame_id = "map"; // 世界坐标系
        
        // 确保使用正确的消息类型，这里使用 utility.hpp 中可能包含的转换函数
        // 如果没有，需要自己实现 cv::Mat 到 Pose 的转换逻辑
        Utility::cvMatToPoseMsg(Twc, pose_msg.pose);
        
        // 3. 发布消息
        m_pose_publisher->publish(pose_msg);
        std::cout << "Published new SLAM pose." << std::endl;
    }
}
