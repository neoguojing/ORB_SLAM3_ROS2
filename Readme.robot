# map (地图坐标系)
## 含义：全局固定坐标系。它是机器人的“世界地图”原点。
## 发布者：通常由 SLAM 节点（如 ORB-SLAM3）发布。
## TODO map -> odom：负责“准不准”（由 ORB-SLAM3 负责）
# odom (里程计坐标系)
## 含义：局部固定坐标系。它记录了机器人从启动那一刻起走了多远
## TODO 发布者：由 EKF 融合节点（robot_localization）或轮式里程计发布。
## odom -> base_link：负责“丝不丝滑”（由 EKF/IMU 负责）。

# base_link (机器人本体坐标系)
## 含义：机器人的物理中心。通常位于机器人旋转轴的中心。

# camera_link (相机坐标系)
## 发布者：由 robot_state_publisher 或 static_transform_publisher 发布。

## base_link -> camera_link：负责“位置对不对”（由你手动测量或标定确定）
