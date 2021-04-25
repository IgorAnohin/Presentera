#!/usr/bin/env python
# coding: utf-8

# # Utilities
import threading
import time
from pathlib import Path
import json


import cv2
import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark

import utils
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dataclasses import dataclass
import os


def to_vector(p1, p2):
    x1, y1 = p1["x"], p1["y"]
    x2, y2 = p2["x"], p2["y"]
    return (x2 - x1), (y2 - y1)


def vec_angle(p1_1, p1_2, p2_1, p2_2):
    x1, y1 = to_vector(p1_1, p1_2)
    x2, y2 = to_vector(p2_1, p2_2)
    return (x1 * x2 + y1 * y2) / (((x1 * x1 + y1 * y1) * (x2 * x2 + y2 * y2)) ** 0.5)


def find_normal(line_point1, line_point2, point):
    x1, y1 = line_point1["x"], line_point1["y"]
    x2, y2 = line_point2["x"], line_point2["y"]
    x3, y3 = point["x"], point["y"]

    dx = x2 - x1
    dy = y2 - y1

    mag = (dx * dx + dy * dy) ** 0.5
    dx /= mag
    dy /= mag

    lambd = (dx * (x3 - x1)) + (dy * (y3 - y1))
    x4 = (dx * lambd) + x1
    y4 = (dy * lambd) + y1
    res = {"x": x4, "y": y4}
    return res

def video_to_images(video_path, out_path):
    """

    :param video_path: path to videofile
    :param out_path: path to output file
    :return: None

    >>> video_to_images('videos/short_next_1forward.mp4', './')
    None
    >>> video_to_images(f"{SHORT_NEXT_4_INCORRECT}.mp4", f"{SHORT_NEXT_4_INCORRECT}")
    None
    """
    os.system(f"mkdir -p {out_path}")

    vidcap = cv.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv.imwrite(f"{out_path}/frame_{count:0>3d}.jpg", image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

# ### Load Images

def similarity_score(pose1, pose2):
    p1 = []
    p2 = []
    pose_1 = np.array(pose1, dtype=np.float)
    pose_2 = np.array(pose2, dtype=np.float)

    # Normalize coordinates
    pose_1[:, 0] = pose_1[:, 0] / max(pose_1[:, 0])
    pose_1[:, 1] = pose_1[:, 1] / max(pose_1[:, 1])
    pose_2[:, 0] = pose_2[:, 0] / max(pose_2[:, 0])
    pose_2[:, 1] = pose_2[:, 1] / max(pose_2[:, 1])

    # L2 Normalization
    #     for joint in range(pose_1.shape[0]):
    #         mag1 = float(math.sqrt(pose_1[joint][0]**2 + pose_1[joint][1]**2))
    #         mag2 = float(math.sqrt(pose_2[joint][0]**2 + pose_2[joint][1]**2))

    #         pose_1[joint][0] = pose_1[joint][0] / mag1
    #         pose_1[joint][1] = pose_1[joint][1] / mag2
    #         pose_2[joint][0] = pose_2[joint][0] / mag2
    #         pose_2[joint][1] = pose_2[joint][1] / mag2

    # Turn (16x2) into (32x1)
    for joint in range(pose_1.shape[0]):
        x1 = pose_1[joint][0]
        y1 = pose_1[joint][1]
        x2 = pose_2[joint][0]
        y2 = pose_2[joint][1]

        p1.append(x1)
        p1.append(y1)
        p2.append(x2)
        p2.append(y2)

    p1 = np.array(p1)
    p2 = np.array(p2)

    # Looking to minimize the distance if there is a match
    # Computing two different distance metrics
    cosine_distance = utils.cosine_distance(p1, p2)
    # weighted_distance = utils.weight_distance(p1, p2, conf1)

    # print("Cosine Distance:", cosine_distance)
    # print("Weighted Distance:", weighted_distance)

    return cosine_distance, 0


# See https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png
BODY_PARTS = {
    "Head": 25,  # Augmented data
    # "Neck": ...,
    # "Neck": 0, # It's a nose actually
    "Neck": 26,  # Augmented data
    "RShoulder": 12,
    "RElbow": 14,
    "RWrist": 16,
    "LShoulder": 11,
    "LElbow": 13,
    "LWrist": 15,
    "RHip": 24,
    # "RKnee": 26, # because we're using simplified model
    # "RAnkle": 28, # because we're using simplified model
    "LHip": 23,
    # "LKnee": 25, # because we're using simplified model
    # "LAnkle": 27 # because we're using simplified model
    # "Chest": ...
    "Chest": 27  # Augmented data
    # "Background": ...
}

POSE_PAIRS = [
    ["Head", "Neck"],
    ["Neck", "RShoulder"],
    ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"],
    ["Neck", "LShoulder"],
    ["LShoulder", "LElbow"],
    ["LElbow", "LWrist"],
    ["Neck", "Chest"],
    ["Chest", "RHip"],
    # ["RHip", "RKnee"],
    # ["RKnee", "RAnkle"],
    ["Chest", "LHip"],
    # ["LHip", "LKnee"],
    # ["LKnee", "LAnkle"]
]

# ### Pose Detector

HEAD = 25
NECK = 26
CHEST = 27


class poseDetector():

    def __init__(self, mode=False, upBody=True, smooth=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)
        self.augmented_landmarks = []
        self.augmented_connections = []

    def _to_point(self, landmark):
        point_dict = { "x": landmark.x, "y": landmark.y}
        if landmark.HasField("visibility"):
            point_dict["visibility"] = landmark.visibility
        return point_dict

    def _point_of(self, x, y, visibility = None):
        point_dict = { "x": x, "y": y}
        if visibility is not None:
            point_dict["visibility"] = visibility
        return point_dict

    def foo(self, landmarks):
        new_landmarks = []
        for landmark in landmarks:
            new_landmarks.append(self._to_point(landmark))

        # --- 25. Head, 26. Neck
        nose = self._to_point(landmarks[PoseLandmark.NOSE])

        left_eye = self._to_point(landmarks[PoseLandmark.LEFT_EYE])
        right_eye = self._to_point(landmarks[PoseLandmark.RIGHT_EYE])

        eye_middle = find_normal(left_eye, right_eye, nose)

        # nose -> eye middle is 2, then half proportion is vector divided by 4
        half_proportion = to_vector(nose, eye_middle)
        half_proportion = (half_proportion[0] / 4, half_proportion[1] / 4)

        # See face proportions https://i.pinimg.com/originals/9d/e1/d1/9de1d16dab7a5d5a819bf351b21ac598.jpg
        # nose -> head top is 5.5 => 11*0.5
        head = self._point_of((nose["x"] + 11 * half_proportion[0]), (nose["y"] + 11 * half_proportion[1]))
        new_landmarks.append(head)

        # neck -> nose is 2.5 => 5*0.5
        neck = self._point_of((nose["x"] - 5 * half_proportion[0]), (nose["y"] - 5 * half_proportion[1]))
        new_landmarks.append(neck)

        # --- 27. Chest
        # chest == (LShoulder + RShoulder) / 2
        lshoulder = self._to_point(landmarks[PoseLandmark.LEFT_SHOULDER])
        rshoulder = self._to_point(landmarks[PoseLandmark.RIGHT_SHOULDER])
        chest = self._point_of((rshoulder["x"] + lshoulder["x"]) / 2, (rshoulder["y"] + lshoulder["y"]) / 2)
        new_landmarks.append(chest)

        return new_landmarks

    def augment_connections(self, connections):
        return frozenset([
            # (HEAD, NECK),
            # (NECK, PoseLandmark.LEFT_SHOULDER),
            # (NECK, PoseLandmark.RIGHT_SHOULDER),
            # (NECK, CHEST),
            # (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.LEFT_SHOULDER),

            (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
            (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),

            (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
            (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),

            # (CHEST, PoseLandmark.LEFT_HIP),
            # (CHEST, PoseLandmark.RIGHT_HIP)
        ])

    def estimate_pose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            self.augmented_landmarks = self.foo(self.results.pose_landmarks.landmark)
            self.augmented_connections = self.augment_connections(self.mpPose.UPPER_BODY_POSE_CONNECTIONS)

            if draw:
                self.mpDraw.draw_landmarks(img, self.augmented_landmarks, self.augmented_connections)
        return img

    def get_landmarks(self):
        return self.augmented_landmarks

    # def get_connections(self, img, draw=True):
    def get_connections(self, img):
        pixel_height, pixel_width, _ = img.shape

        self.connections_list = []

        landmarks = self.get_landmarks()
        connections = self.augmented_connections
        debug = []
        if landmarks and connections:
            for (first_none_idx, second_node_idx) in connections:
                first_node, second_node = landmarks[first_none_idx], landmarks[second_node_idx]
                if first_node["visibility"] < 0.6 or second_node["visibility"] < 0.6:
                    self.connections_list.append(None)
                    self.connections_list.append(None)

                    continue
                debug.append((landmarks, first_node["visibility"], second_node["visibility"]))
                first_node, second_node = (first_node["x"], first_node["y"]), (second_node["x"], second_node["y"])

                first_node = (pixel_width*first_node[0], pixel_height*first_node[1])
                second_node = (pixel_width*second_node[0], pixel_height*second_node[1])

                self.connections_list.append(first_node)
                self.connections_list.append(second_node)

            # for id, lm in enumerate(self.augmented_landmarks):
            # for id, lm in enumerate(self.results.pose_landmarks.landmark):
            #     pixel_height, pixel_width, c = img.shape
            #     # print(id, lm)
            #     cx, cy = int(lm.x * pixel_width), int(lm.y * pixel_height)
            #     self.connections_list.append([id, cx, cy])
            #     if draw:
            #         cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.connections_list


# ### Sample

def crop_and_resize_matching(connections1, connections2):
    missed_keys = 0
    total_keys = len(connections1)

    con1_without_none = []
    con2_without_none = []
    for con1, con2 in zip(connections1, connections2):
        if con1 is None or con2 is None:
            missed_keys += 1
            continue
        con1_without_none.append(con1)
        con2_without_none.append(con2)

    connections1_new = np.array(con1_without_none)
    connections2_new = np.array(con2_without_none)

    if not con1_without_none:
        return (connections1_new, connections2_new, (0, 0))

    connections1_new[:, 0] = connections1_new[:, 0] - min(connections1_new[:, 0])
    connections1_new[:, 1] = connections1_new[:, 1] - min(connections1_new[:, 1])

    connections2_new[:, 0] = connections2_new[:, 0] - min(connections2_new[:, 0])
    connections2_new[:, 1] = connections2_new[:, 1] - min(connections2_new[:, 1])

    resize_x = max(connections2_new[:, 0]) / max(connections1_new[:, 0])
    resize_y = max(connections2_new[:, 1]) / max(connections1_new[:, 1])

    connections1_new[:, 0] = connections1_new[:, 0] * resize_x
    connections1_new[:, 1] = connections1_new[:, 1] * resize_y

    cosine_score, weighted_score = similarity_score(connections1_new, connections2_new)
    bounty = (total_keys - missed_keys) / total_keys
    cosine_score *= bounty
    weighted_score *= bounty
    return (connections1_new, connections2_new, (cosine_score, weighted_score))


def match_frames_window(sample_frame_window, real_frame_window, sample_connections_window, real_connections_window, window_filter=None) -> float:
    sample_length = len(sample_frame_window)
    real_length = len(real_frame_window)

    if real_length < sample_length:
        # sample_window = sample_window[:real_length]
        sample_connections_window = sample_connections_window[-real_length:]
    elif real_length > sample_length:
        # real_window = real_window[-real_window:]
        real_connections_window = real_connections_window[-sample_length:]

    length = min(sample_length, real_length)
    del sample_length, real_length

    score_index = 0
    scores = np.zeros(length, dtype=float)
    # print("Start")
    for (sample_pose, real_pose) in zip(sample_connections_window, real_connections_window):
        if (sample_pose is None) or (real_pose is None):
            score_index += 1
            continue

        (_, _, (cosine_distance, weighted_distance)) = crop_and_resize_matching(sample_pose, real_pose)
        # print("Cosine", cosine_distance)
        scores[score_index] = cosine_distance
        score_index += 1

    # print("Scores:", scores)
    final_score = np.sum(scores) / length  # here probably could be amount of successful additions in loop
    # print("Stop. Final:", final_score)
    return final_score


def initialize_sample(sample_frame_paths: list, window: int) -> tuple:
    sample_frame_window = np.array([cv.imread(str(pic_path)) for pic_path in sample_frame_paths])
    sample_connections_window = []

    detector = poseDetector()

    sample_index = 0
    for frame in sample_frame_window[-window:]:
        if sample_index >= window:
            break

        frame = detector.estimate_pose(frame)
        connections_list = detector.get_connections(frame)
        if len(connections_list) == 0:
            connections_list = None

        sample_connections_window.append(connections_list)
        sample_index += 1

    return sample_frame_window, sample_connections_window


cap = cv.VideoCapture(0)


def prepare_initial_real_frame(detector, window: int):
    real_frame_window = np.repeat(None, window)
    real_connections_window = []
    for i in range(window):
        success, frame = cap.read()
        if not success:
            continue

        frame = detector.estimate_pose(frame)
        connections_list = detector.get_connections(frame)

        if len(connections_list) == 0:
            connections_list = None

        real_frame_window[i] = cap.read()
        real_connections_window.append(connections_list)
    return real_frame_window, real_connections_window

# TEST VIDEO

VIDEOS_FOLDER = Path("./videos")
# SHORT_NEXT_4_INCORRECT = VIDEOS_FOLDER / "short_next_4incorrect"

SHORT_NEXT_1FORWARD = VIDEOS_FOLDER / "short_next_1forward"
SHORT_NEXT_1LEFT = VIDEOS_FOLDER / "short_next_2left"
BACK = VIDEOS_FOLDER / "slides-back"

FRAME_PATTERN = "frame_*.jpg"


@dataclass()
class Gesture:
    action: str
    sample_frame_window: list
    sample_connections_window: list


DEBUG = True


def infinity_worker(d, return_value: bool = False):
    detector = poseDetector()

    if DEBUG:
        for video_name in d:
            path = video_name
            print("Loading image:", path)
            video_to_images(f"{path}.mp4", path)

    sample_frames_lists = []
    for video_name in d:
        path = Path(video_name)
        sample_frame_paths = list(sorted(path.glob(FRAME_PATTERN)))
        window_size = min(15, len(sample_frame_paths))

        sample_frame_window, sample_connections_window = initialize_sample(sample_frame_paths, window_size)
        gesture = Gesture(video_name, sample_frame_window, sample_connections_window)

        sample_frames_lists.append(gesture)

    real_frame_window, real_connections_window = prepare_initial_real_frame(detector, window_size)

    previous_action_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            continue
        frame = detector.estimate_pose(frame)
        connections_list = detector.get_connections(frame)
        if len(connections_list) == 0:
            connections_list = None

        np.roll(real_frame_window, -1)
        real_frame_window[-1] = frame

        real_connections_window[:-1] = real_connections_window[1:]
        real_connections_window[-1] = connections_list

        for gesture in sample_frames_lists:
            window_score = match_frames_window(gesture.sample_frame_window, real_frame_window,
                                               gesture.sample_connections_window, real_connections_window)
            data: PlotData = d[gesture.action]
            if data.xs:
                data.xs.append(data.xs[-1] + 1)
            else:
                data.xs.append(0)
            data.ys.append(window_score)

            if len(data.ys) > 100:
                # print(f"Score {window_score} previous {data.ys[-10]}")

                last_100 = data.ys[-100:]
                avg_y = sum(last_100) / 100
                max_y = max(last_100)
                min_y = min(last_100)
                dist = max_y = min_y
                global_max_y = max(data.ys)
                if dist > 0.02  and avg_y > 0.8 and window_score > 0.95 and \
                        window_score > (global_max_y * 0.99) and window_score > min(avg_y * 1.04, max_y):
                    print(f"Window triggered with score: {window_score} for {gesture.action}. Max {global_max_y}, AVG: {avg_y}")
                    if time.time() - previous_action_time > 5:
                        previous_action_time = time.time()
                        yield gesture.action


        cv2.imshow("Image", frame)
        cv2.waitKey(1)


@dataclass
class PlotData:
    xs: list
    ys: list


def animate(i, ax, data: PlotData):
    xs = data.xs[-505:]
    ys = data.ys[-505:]
    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('Actions')


import pyautogui, time


def main_without_plotting():
    config_file = "./configuretion.json"

    data_dict = dict()
    video_to_action_dict = {}

    with open(config_file) as config_file:
        config = json.load(config_file)
        print(config)
        for entry in config:
            video = entry["src"].replace(".mp4", "")
            data_dict[video] = PlotData([], [])
            video_to_action_dict[video] = entry["action"]

    for video_triggered in infinity_worker(data_dict, True):
        print("Detected action:", video_triggered)
        if video_to_action_dict[video_triggered] == "Следующий слайд":
            # pyautogui.keyDown('alt')
            # time.sleep(.2)
            # pyautogui.press('tab')
            # time.sleep(.2)
            # pyautogui.keyUp('alt')

            pyautogui.press('right')
        elif video_to_action_dict[video_triggered] == "Предыдущий слайд":
            pyautogui.press('left')
        else:
            print("Nothing")


def test(data_dict):
    for action in infinity_worker(data_dict):
        print("Action:", action)


def main():
    config_file = "./configuretion.json"

    data_dict = dict()
    video_to_action_dict = {}

    anims = []
    figs = []

    with open(config_file) as config_file:
        config = json.load(config_file)
        print(config)
        for entry in config:
            video = entry["src"].replace(".mp4", "")
            data_dict[video] = PlotData([], [])
            video_to_action_dict[video] = entry["action"]

            fig = plt.figure()
            figs.append(fig)
            ax = fig.add_subplot(1, 1, 1)
            an = animation.FuncAnimation(fig, animate, fargs=(ax, data_dict[video]), interval=100)
            anims.append(an)

    threading.Thread(target=test, args=(data_dict,)).start()
    plt.show()


if __name__ == "__main__":
    main()
    # main_without_plotting()
