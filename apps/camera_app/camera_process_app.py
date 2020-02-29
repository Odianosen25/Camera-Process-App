import adbase as ad
import numpy as np
import imutils
import cv2
import traceback
import threading
import json
import os
import time


__VERSION__ = "0.1"


class CameraProcessApp(ad.ADBase):
    def initialize(self):
        self.adapi = self.get_ad_api()
        self.mqtt = self.get_plugin_api("MQTT")

        self.lock = threading.Lock()

        self._capturing = False
        self._camera_capturing = None
        self._video_processing = None
        self._video_capture = None
        self._image_data = None
        self._last_motion_state = None
        self._last_reported_time = self.adapi.get_now_ts()
        self.camera_fps = self.args.get("frames_per_second", 20)
        self.location = self.args.get("location", self.name)

        # for motion detection
        self.motion_total_frame = 0
        self.motion_bg = None
        self.motion_previous_frame = None

        # setup to be ran in a thread, to avoid holding up AD
        self.adapi.run_in(self.setup_video_capture, 0)

        # register web app
        self.adapi.register_route(self.process_stream)

    def setup_video_capture(self, kwargs):
        """This sets up the Video Capture instance"""
        file_location = os.path.dirname(os.path.abspath(__file__))

        # load caffee object detection classes
        labelsPath = self.args.get("caffee_labels", f"{file_location}/caffee.names")

        # load object detection models
        object_prototxt = self.args.get(
            "object_prototxt", f"{file_location}/MobileNetSSD_deploy.prototxt.txt"
        )
        object_model = self.args.get(
            "object_model", f"{file_location}/MobileNetSSD_deploy.caffemodel"
        )
        self._camera_url = self.args.get("camera_url")
        self._topic = f"camera/{self.location}"

        if self._camera_url == None:
            raise ValueError("Camera URL not provided")

        try:
            # setup caffee object net
            self.caffee_object_net = cv2.dnn.readNetFromCaffe(
                object_prototxt, object_model
            )
            self.caffee_classes = open(labelsPath).read().strip().split("\n")

            # now setup capture
            # get capture
            self._video_capture = cv2.VideoCapture(self._camera_url)

            # get height and width
            height = self.args.get("height", 720)
            width = self.args.get("width", 1280)

            # set height and width
            self._video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            self._video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        except:
            if self._video_capture != None:
                self._video_capture.release()
                self._video_capture = None

            self.adapi.error(traceback.format_exc(), level="ERROR")

        if self._video_capture != None and self._video_capture.isOpened():
            # at this point, the video capture has been instanciated

            self._capturing = True
            # start processing
            self._camera_capturing = self.adapi.create_task(self.camera_capturing())

    async def camera_capturing(self):
        """This processes the Video Capturing"""
        self.adapi.log("Starting video processing and Motion Detection")

        while self._capturing:
            try:
                # Check success
                if self._video_capture != None:
                    if not self._video_capture.isOpened():
                        raise Exception("Video device is not opened")

                    capture, frame = await self.adapi.run_in_executor(
                        self._video_capture.read
                    )

                    if capture:
                        with self.lock:
                            self._image_data = frame

                        # check for motion detection
                        await self.adapi.run_in_executor(self.motion_update, frame)

            except:
                self.adapi.error(
                    "There was an error when processing image capture", level="ERROR"
                )
                self.adapi.error(traceback.format_exc(), level="ERROR")
                await self.adapi.sleep(5)

    def motion_update(self, frame):
        """To update the motion detection frame"""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (7, 7), 0)

        if self.motion_previous_frame is None:
            self.motion_previous_frame = image.copy()
            self.motion_time_frame = time.time()

        if self.motion_total_frame > int(self.camera_fps / 2):
            # the number of frames collected, more than required
            # detect motion in the image

            if (time.time() - self.motion_time_frame) > 10:
                # meaning every 10 seconds, it must load up a previous frame for slow motion
                # update the background model by accumulating the weighted average
                cv2.accumulateWeighted(self.motion_previous_frame, self.motion_bg, 0.4)
                self.motion_previous_frame = None

            # run motion detection
            self.motion_detect(image)
            self.motion_total_frame = 0

        else:
            # if the background model is None, initialize it
            if self.motion_bg is None:
                self.motion_bg = image.copy().astype("float")
                return

            # update the background model by accumulating the weighted
            # average
            cv2.accumulateWeighted(image, self.motion_bg, 0.4)
            self.motion_total_frame += 1

    def motion_detect(self, image):
        """To process motion detection"""
        tVal = 25
        # compute the absolute difference between the background model
        # and the image passed in, then threshold the delta image
        delta = cv2.absdiff(self.motion_bg.astype("uint8"), image)
        thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]

        # perform a series of erosions and dilations to remove small
        # blobs
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in the thresholded image and initialize the
        # minimum and maximum bounding box regions for motion
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = imutils.grab_contours(cnts)
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)

        # if no contours were found, return None so no motion
        if len(cnts) > 0 and self._last_motion_state != "on":
            # motion was detected
            self.mqtt.mqtt_publish(self._topic, json.dumps({"motion_detected": "on"}))
            self._last_motion_state = "on"

            if self.args.get("detect_objects") is True:
                self.adapi.run_in(self.detect_objects, 0, image_data=image)

        elif self._last_motion_state != "off":
            # motion was not detected
            self.mqtt.mqtt_publish(self._topic, json.dumps({"motion_detected": "off"}))
            self._last_motion_state = "off"

    def detect_objects(self, kwargs):
        image_data = kwargs["image_data"]
        minimum_confidence = self.args.get("minimum_confidence", 0.4)

        # generate random colours for each object
        colours = np.random.uniform(0, 255, size=(len(self.caffee_classes), 3))
        image_data = imutils.resize(image_data, width=400)

        try:
            (h, w) = image_data.shape[:2]
        except TypeError as t:
            self.adapi.error(t)
            return None

        blob = cv2.dnn.blobFromImage(
            cv2.resize(image_data, (300, 300)), 0.007843, (300, 300), 127.5
        )

        # pass the blob through the network and obtain the detections and
        # predictions
        self.caffee_object_net.setInput(blob)
        detections = self.caffee_object_net.forward()

        # loop over the detections
        number = 0
        detected_objects = {}

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > minimum_confidence:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                obj_class = self.caffee_classes[idx]

                if obj_class == "person":
                    obj_class = f"person_{number}"

                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    obj_box = {}
                    obj_box["startX"] = int(startX)
                    obj_box["startY"] = int(startY)
                    obj_box["endX"] = int(endX)
                    obj_box["endY"] = int(endY)
                    obj_box["confidence"] = float(confidence)
                    detected_objects[obj_class] = obj_box
                    number += 1

        self.mqtt.mqtt_publish(self._topic, json.dumps(detected_objects))

    async def process_stream(self, request):
        """This is for the Web Stream for the MJPEG"""
        stream = web.StreamResponse(
            status=200, reason="OK", headers={"Content-Type": "text/html"}
        )

        stream.content_type = "multipart/x-mixed-replace; boundary=frame"

        await stream.prepare(request)

        while True:
            if request.transport.is_closing():
                break

            encodedImage = await self.adapi.run_in_executor(self.get_image_data)

            if encodedImage is None:
                break

            await stream.write(
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + encodedImage + b"\r\n"
            )
            await self.adapi.sleep(0.1)

        await stream.write_eof()
        return stream

    def get_image_data(self, asbytes=True):
        image_data = None
        with self.lock:
            if self._image_data is not None:

                try:
                    image_data = self._image_data.copy()

                except ValueError as v:
                    self.adapi.error(v, level="ERROR")

        data = None
        try:
            capture, image_data = cv2.imencode(".jpg", image_data)

            if capture is True:
                if asbytes is True:
                    data = image_data.tobytes()
                else:
                    data = image_data

        except Exception:
            self.adapi.error(traceback.format_exc(), level="ERROR")

        return data

    async def terminate(self):
        self.adapi.log("Stopping camera Video Capturing")

        if self._capturing is True:
            self._capturing = False

            await self.adapi.sleep(1)

        if self._camera_capturing != None and (
            not self._camera_capturing.done() or not self._camera_capturing.cancelled()
        ):
            self.adapi.log("Cancelling video processing and capturing")
            try:
                self._camera_capturing.cancel()

            except asyncio.CancelledError:
                self.adapi.error(
                    "Cancelling video processing and capturing", level="DEBUG"
                )

            self._camera_capturing = None

        if self._video_capture != None:  # first ensure the capture is closed
            self.adapi.log(
                f"Video Capture has been stopped. Releasing video capture now"
            )
            await self.adapi.run_in_executor(self._video_capture.release)
            self._video_capture = None
            await self.adapi.sleep(2)

        # reset motion detect
        await self.mqtt.mqtt_publish(self._topic, json.dumps({"motion_detcted": "off"}))
        self.motion_total_frame = 0
        self.motion_bg = None
