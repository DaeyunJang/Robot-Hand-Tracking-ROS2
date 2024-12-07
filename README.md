# Robot-Hand-Tracking-ROS2
Hand tracking using camera and 6-axis robot arm with ROS2

## Demo with the simulation
#### Visual servoing
Test on Doosan Robot simulation
![Demo gif](https://github.com/DaeyunJang/Robot-Hand-Tracking-ROS2/raw/humble/media/visual_servoing_test.gif)


---
## Index
1. [Prerequisite](#prerequisite)
2. [System Requirements](#system-requirements)
4. [Installation](#installation)
6. [Node & Topic](#node--topic)
9. [Execute](#execute)

## Prerequisite
#### OS
Ubuntu 22.04.6 LTS
#### Software
- ROS2 humble
  <https://docs.ros.org/en/humble/index.html>

## System Requirements
- Ubuntu 22.04 (ROS2 humble) or Ubuntu 20.04 (ROS2 foxy)
- Realsense camera (D415)

## Installation
##### python
```bash
  $ pip install pyserial mediapipe pyrealsense2
```

## Node & Topic
- **'handpose_node'**
  - Publisher<br/>
    `mediapipe_hand_landmarks`<br/>
    `camera_to_hand_vector`<br/>

## Execute
#### Clone repository and move to this project
```
git clone https://github.com/DaeyunJang/Robot-Hand-Tracking-ROS2.git
cd Robot-Hand-Tracking-ROS2
```

#### Build using colcon
```
colcon build --symlink-install
```

#### Source setup file
```
. install/setup.bash
```

#### Run
```
ros2 run handpose_pkg handpose_publisher
ros2 topic echo /camera_to_hand_vector
```

## Execute (python)
You can run the code without ROS2 system.
It will calculate the center of the hand palm and its normal vector. This position and normal vector can change to 6D coordinate [x,y,z,R,P,Y].
Finally these data will send through the TCP communication.

```
cd {PROJECT_DIRECTORY}/scripts
```
```
python3 main.py
```
#### main.py
These codes consist of 3 processes
1. Find the hand landmarks and calculate the palm center and its `normal vector` in the real-world.
2. Show plot using the hand landmarks.
3. Send the `target pose of the end-effector` as 6D Euler [x,y,z,R,P,Y] othrough the TCP comminication. 




## Note
Imagine, you have the 6D robot arm and the camera mounted on the end-effector of this robot.
This code can find the hand pose using Google Mediapipe, and Realsense camera will calculate the real-world [x,y,z] of 'finger tip'. Finally, ROS2 topic will publish these coordinate values.
-> As you can expect, [x,y,z] means the distance from the center of the camera

I implemented that this ROS system send(publish) the topic 'camera_to_hand_vector' as real-world [x,y,z] of 'finger tip'. (handpose_indexfinger_publisher.py)

If you want to move the 6D robot arm in the simulation or real-world,
you can subscribe 'camera_to_hand_vector' and move relative its end-effector by [x, y, z].

If you want to publish all hand pose data,
you will develop the code. (the file 'handpose_publisher.py' in src/handpose_pkg/handpose_pkg is not applied ROS2 system)

### Debug
Please check the camera depth unit option(line 171 of 'handpose_indexfinger_publisher.py') because the unit(mm) can be different for each model.
If you don't use the Realsense camera, you have to modify the code for capturing the camera frame.

