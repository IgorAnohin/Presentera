#!/usr/bin/env python
# coding: utf-8

# # Utilities

# In[25]:
import threading


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

    #     print("normal:", f"x1,y1: {line_point1}, x2,y2: {line_point2}", f"x3,y3: {point}", f"mag: {mag}, lambd: {lambd}", f"res: {res}")
    return res


# In[2]:


def video_to_images(video_path, out_path):
    vidcap = cv.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv.imwrite(f"{out_path}/frame_{count:0>3d}.jpg", image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


# video_to_images('videos/short_next_1forward.mp4', './')
# video_to_images(f"{SHORT_NEXT_4_INCORRECT}.mp4", f"{SHORT_NEXT_4_INCORRECT}")


# In[4]:


import math
import time
from pathlib import Path

import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark

import utils

# In[5]:


VIDEOS_FOLDER = Path("./videos")

SHORT_NEXT_1FORWARD = VIDEOS_FOLDER / "short_next_1forward"
SHORT_NEXT_1LEFT = VIDEOS_FOLDER / "short_next_2left"
SHORT_NEXT_4_INCORRECT = VIDEOS_FOLDER / "short_next_4incorrect"

FRAME_PATTERN = "frame_*.jpg"

# ### Load Images

# frame1 = cv.imread(f"images/running1.jpg") # input
frame1 = cv.imread(str(SHORT_NEXT_1FORWARD / "frame_000.jpg"))

# frame1 = cv.imread(f"images/running2.jpg") # output
frame2 = cv.imread(str(SHORT_NEXT_1LEFT / "frame_000.jpg"))  # test succ
# frame2 = cv.imread(SHORT_NEXT_1LEFT / "frame_020.jpg") # test fail
# frame2 = cv.imread(SHORT_NEXT_1LEFT / "frame_003.jpg") # test full fail

# Ensure images are the same size
hi = min(frame1.shape[0], frame2.shape[0])
wi = min(frame1.shape[1], frame2.shape[1])

frame1 = cv.resize(frame1, (wi, hi))
frame2 = cv.resize(frame2, (wi, hi))

# In[7]:
frameWidth = frame1.shape[1]
frameHeight = frame1.shape[0]

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
    weighted_distance = utils.weight_distance(p1, p2, conf1)

    # print("Cosine Distance:", cosine_distance)
    # print("Weighted Distance:", weighted_distance)

    return cosine_distance, weighted_distance


# In[32]:


def visualize_output(pose1, pose2, size):
    assert (len(pose1) == len(pose2))
    pose_len = len(pose1)

    # Initialize blank canvas
    canvas = np.ones(size)

    # Plot points on images
    for pair in POSE_PAIRS:
        part_from = pair[0]
        part_to = pair[1]
        assert (part_from in BODY_PARTS)
        assert (part_to in BODY_PARTS)

        id_from = BODY_PARTS[part_from]
        id_to = BODY_PARTS[part_to]

        if id_from >= pose_len:
            continue
        elif id_to >= pose_len:
            continue

        if pose1[id_from] and pose1[id_to]:
            cv.line(canvas, pose1[id_from], pose1[id_to], (0, 255, 0), 3)
            cv.ellipse(canvas, pose1[id_from], (4, 4), 0, 0, 360, (0, 255, 0), cv.FILLED)
            cv.ellipse(canvas, pose1[id_to], (4, 4), 0, 0, 360, (0, 255, 0), cv.FILLED)

    for pair in POSE_PAIRS:
        part_from = pair[0]
        part_to = pair[1]
        assert (part_from in BODY_PARTS)
        assert (part_to in BODY_PARTS)

        id_from = BODY_PARTS[part_from]
        id_to = BODY_PARTS[part_to]

        if id_from >= pose_len:
            continue
        elif id_to >= pose_len:
            continue

        if pose2[id_from] and pose2[id_to]:
            cv.line(canvas, pose2[id_from], pose2[id_to], (255, 0, 0), 3)
            cv.ellipse(canvas, pose2[id_from], (4, 4), 0, 0, 360, (255, 0, 0), cv.FILLED)
            cv.ellipse(canvas, pose2[id_to], (4, 4), 0, 0, 360, (255, 0, 0), cv.FILLED)

    # Visualize images
    fig3 = plt.figure(figsize=(10, 10))
    plt.imshow(canvas[:, :, ::-1])
    plt.grid(True)


# ## Solution 1

# In[9]:


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

# In[26]:


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

    def augment_landmarks(self, landmarks):
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
        # return frozenset.union(connections, frozenset([
        #     (HEAD, NECK),
        #     (NECK, PoseLandmark.LEFT_SHOULDER),
        #     (NECK, PoseLandmark.RIGHT_SHOULDER),
        #     (NECK, CHEST),
        #     (CHEST, PoseLandmark.LEFT_HIP),
        #     (CHEST, PoseLandmark.RIGHT_HIP)
        # ]))
        return frozenset([
            # (HEAD, NECK),
            # (NECK, PoseLandmark.LEFT_SHOULDER),
            # (NECK, PoseLandmark.RIGHT_SHOULDER),
            # (NECK, CHEST),
            (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.LEFT_SHOULDER),

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
            self.augmented_landmarks = self.augment_landmarks(self.results.pose_landmarks.landmark)
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
        if landmarks and connections:
            for (first_none_idx, second_node_idx) in connections:
                first_node, second_node = landmarks[first_none_idx], landmarks[second_node_idx]
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

# In[47]:


detector = poseDetector()

# In[48]:


frame1 = detector.estimate_pose(frame1)
connections1 = detector.get_connections(frame1)

# In[49]:


frame2 = detector.estimate_pose(frame2)
connections2 = detector.get_connections(frame2)

conf1 = [1.0 for x in connections1]

# In[52]:


# plt.figure(figsize=(15, 15))
# plt.imshow(frame1)


def crop_and_resize_matching(connections1, connections2):
    connections1_new = np.array(connections1)
    connections2_new = np.array(connections2)

    connections1_new[:, 0] = connections1_new[:, 0] - min(connections1_new[:, 0])
    connections1_new[:, 1] = connections1_new[:, 1] - min(connections1_new[:, 1])

    connections2_new[:, 0] = connections2_new[:, 0] - min(connections2_new[:, 0])
    connections2_new[:, 1] = connections2_new[:, 1] - min(connections2_new[:, 1])

    resize_x = max(connections2_new[:, 0]) / max(connections1_new[:, 0])
    resize_y = max(connections2_new[:, 1]) / max(connections1_new[:, 1])

    connections1_new[:, 0] = connections1_new[:, 0] * resize_x
    connections1_new[:, 1] = connections1_new[:, 1] * resize_y

    return (connections1_new, connections2_new, similarity_score(connections1_new, connections2_new))


# In[54]:


connections1_new, connections2_new, score = crop_and_resize_matching(connections1, connections2)

# Compute the new similarity scores

# Visualize the result

# In[55]:


pose1_resized = tuple(map(tuple, connections1_new))
pose2_resized = tuple(map(tuple, connections2_new))

# Get dimensions of output window
connections1 = np.array(connections1_new)
connections2 = np.array(connections2_new)
max_y = max(max(connections1[:, 0]), max(connections2[:, 0]))
max_x = max(max(connections1[:, 1]), max(connections2[:, 1]))
dim = (max_x, max_y, 3)
#TODO visualize_output(pose1_resized, pose2_resized, dim)


# ## Camera

# In[56]:


# expected_pose = [cv.imread(str(pic_path)) for pic_path in list(sorted(SHORT_NEXT_1FORWARD.glob(FRAME_PATTERN)))]


# In[64]:

def match_frames_window(sample_frame_window, real_frame_window, sample_connections_window, real_connections_window, window_filter=None) -> float:
    sample_length = len(sample_frame_window)
    real_length = len(real_frame_window)

    if real_length < sample_length:
        # sample_window = sample_window[:real_length]
        sample_connections_window = sample_connections_window[:real_length]
    elif real_length > sample_length:
        # real_window = real_window[-real_window:]
        real_connections_window = real_connections_window[-real_frame_window:]

    length = min(sample_length, real_length)
    del sample_length, real_length

    score_index = 0
    scores = np.zeros(length, dtype=float)
    for (sample_pose, real_pose) in zip(sample_connections_window, real_connections_window):
        if (sample_pose is None) or (real_pose is None):
            score_index += 1
            continue

        (_, _, (cosine_distance, weighted_distance)) = crop_and_resize_matching(sample_pose, real_pose)
        scores[score_index] = cosine_distance
        score_index += 1

    # print("Scores:", scores)
    final_score = np.sum(scores) / length  # here probably could be amount of successful additions in loop
    return final_score


# TEST VIDEO

SAMPLE_FRAME_PATHS = list(sorted(SHORT_NEXT_1FORWARD.glob(FRAME_PATTERN)))
WINDOW_SIZE = min(15, len(SAMPLE_FRAME_PATHS))

sample_frame_window = np.array([cv.imread(str(pic_path)) for pic_path in SAMPLE_FRAME_PATHS])
sample_connections_window = []

detector = poseDetector()

sample_index = 0
for frame in sample_frame_window:
    if sample_index >= WINDOW_SIZE:
        break

    frame = detector.estimate_pose(frame)
    connections_list = detector.get_connections(frame)
    if len(connections_list) == 0:
        connections_list = None

    sample_connections_window.append(connections_list)
    sample_index += 1

cap = cv.VideoCapture(0)
detector = poseDetector()

overall_sum = 0.0
amount = 0

real_frame_window = np.repeat(None, WINDOW_SIZE)
real_connections_window = []
# real_poses = np.repeat(None, WINDOW_SIZE, axis=0)
for i in range(WINDOW_SIZE):
    success, frame = cap.read()
    if not success:
        continue

    frame = detector.estimate_pose(frame)
    connections_list = detector.get_connections(frame)

    if len(connections_list) == 0:
        connections_list = None

    real_frame_window[i] = cap.read()
    real_connections_window.append(connections_list)


# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []

import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random


# This function is called periodically from FuncAnimation
def animate(i, xs, ys):

    # Read temperature (Celsius) from TMP102
    temp_c = round(random.randint(1, 100), 2)

    # Add x and y to lists
    # xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    # ys.append(temp_c)

    xs = xs[:105]
    ys = ys[:105]
    # Draw x and y lists
    ax.clear()
    ax.plot(xs,      ys)

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('TMP102 Temperature over Time')
    plt.ylabel('Temperature (deg C)')


# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=100)

threading.Thread(target=plt.show).start()
# plt.show()
while True:
    start_time = time.time()

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

    window_score = match_frames_window(sample_frame_window, real_frame_window, sample_connections_window, real_connections_window)
    print("Window score:", window_score)

    end_time = time.time()

    # Add x and y to lists
    xs.append(dt.datetime.now().strftime('%M:%S.%f'))
    ys.append(window_score)

    cv2.imshow("Image", frame)
    cv2.waitKey(1)

    amount += 1
    diff = (end_time - start_time)
    fps = int(round(1.0 / diff))
    overall_sum += fps
    # print(f"{1000 * diff}ms | {fps}fps | {overall_sum / amount} avg fps")
