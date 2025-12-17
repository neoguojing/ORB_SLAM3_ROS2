# 设置变量 (请根据实际路径修改，或者通过环境变量传入)
ORB_ROOT ?= $(ORBSLAM3_ROOT_DIR)
WS_PATH   = $(ORB_ROOT)/../orbslam3
VOC_PATH  = $(ORB_ROOT)/Vocabulary/ORBvoc.txt
YAML_PATH = $(ORB_ROOT)/Examples/Monocular/TUM1.yaml

# 默认指令：编译
all: build

# 1. 编译项目
build:
	@echo "开始编译 ROS 2 节点..."
	cd $(WS_PATH) && colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	@echo "编译完成。"

# 2. 运行单目节点
# 使用 bash -c 来确保 source 命令在当前 shell 进程中生效
run_mono:
	@echo "启动 ORB-SLAM3 单目节点..."
	bash -c "source $(WS_PATH)/install/setup.bash && \
	ros2 run orbslam3 mono $(VOC_PATH) $(YAML_PATH)"

# 3. 清理编译产物
clean:
	@echo "清理 build, install, log 目录..."
	rm -rf $(WS_PATH)/build $(WS_PATH)/install $(WS_PATH)/log

# 4. 辅助指令：查看当前配置
info:
	@echo "ORB_ROOT: $(ORB_ROOT)"
	@echo "Workspace: $(WS_PATH)"
	@echo "Vocabulary: $(VOC_PATH)"
	@echo "Config YAML: $(YAML_PATH)"

.PHONY: all build run_mono clean info