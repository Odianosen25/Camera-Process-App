# Camera-Process-App
AppDaemon App used to process RTSP or USB Video from Cameras, for security purposes

This is a <b>WORK IN PROGRESS</b>

When using automation hubs like Home Assistant (HA), and there is the need to integrate a RTSP stream, it is usually not possible to do it without an external software like motionEye. This app is designed to give users the ability to process RTSP cameras right within AppDaemon, without the need of using such external softwares. When it detects motion or objects, the data is forwarded to a MQTT topic `camera/<location>`. The advantage of using this above softwares like motionEye are as follows:
- Compared to MotionEye, this app uses way less resources to function
- Once complete, this app will have way more features than the likes of MotionEye (please see below)
- The app allow for better flexibility, and integration with Home Assistant

## Example Configuration below
```yaml
hallway_camera:
  module: camera_process_app
  class: CameraProcessApp
  camera_url: rtsp://freja.hiof.no:1935/rtplive/definst/hessdalen03.stream
  height: 1080 # default 720
  width: 1920 # default 1280
  frames_per_second: 25 # default 20
  location: Hallway
  detect_objects: True
```

When the app is setup and all running, the MJPEG can be accessed using the following url `http://ADIP:ADPORT/app/<camera_app_name>`. This app uses the internal port number of AD, so there is no need of making use of multiple ports on the PC its running. 

The app also processes object detection using the Caffee Model, but this will be upgraded to use the Yolo dataset over time, for more accurate object detection. The object detection will only be processed, when it detects motion only.

## TODO
* [x] Stream over MJPEG
* [x] Process motion detection
* [] Use Yolo instead of Caffee for object detection
* [] Add options to limit what objects to focus on
* [] Add Facial Recognition
* [] Add Object Tracking
