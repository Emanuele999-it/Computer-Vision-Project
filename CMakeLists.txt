cmake_minimum_required(VERSION 2.8)

# Set the project name
project(CV-Project)

# Find OpenCV library
find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

# Create executables
add_executable(CVProject src/main.cpp src/imageProcessing.cpp src/displayMat.cpp 
                src/fieldSegmentation.cpp src/colorSuppression.cpp src/playerSegmentation.cpp
                src/mostFrequentColorFiltering.cpp src/playerDetection.cpp src/playerTeam.cpp)

add_executable(TrainModel src/trainModel.cpp)

# Link against OpenCV library
target_link_libraries(CVProject ${OpenCV_LIBS})

target_link_libraries(TrainModel ${OpenCV_LIBS})