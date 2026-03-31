cmake_minimum_required(VERSION 3.16)
project(ros2_fortress_ws)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_python REQUIRED)
find_package(ament_cmake REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  ros2_fortress_ws/pose_frame_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install worlds
install(DIRECTORY worlds/
  DESTINATION share/${PROJECT_NAME}/worlds
)

# Install models
install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
