from dataclasses import dataclass
from enum import IntEnum
from typing import List

from mediapipe.python.solutions.pose import PoseLandmark
import mediapipe as mp
import cv2 as cv

import drawing_utils


@dataclass
class VisiblePoint2D:
    x: float
    y: float
    visibility: float = 1.0


def to_vector(p1: VisiblePoint2D, p2: VisiblePoint2D) -> VisiblePoint2D:
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    return VisiblePoint2D((x2 - x1), (y2 - y1))


def vec_angle(p1_1: VisiblePoint2D, p1_2: VisiblePoint2D, p2_1: VisiblePoint2D, p2_2: VisiblePoint2D):
    vec1, vec2 = to_vector(p1_1, p1_2), to_vector(p2_1, p2_2)
    x1, y1 = vec1.x, vec1.y
    x2, y2 = vec2.x, vec2.y

    return (x1 * x2 + y1 * y2) / (((x1 * x1 + y1 * y1) * (x2 * x2 + y2 * y2)) ** 0.5)


def find_normal(line_point1: VisiblePoint2D, line_point2: VisiblePoint2D, point: VisiblePoint2D) -> VisiblePoint2D:
    x1, y1 = line_point1.x, line_point1.y
    x2, y2 = line_point2.x, line_point2.y
    x3, y3 = point.x, point.y

    dx = x2 - x1
    dy = y2 - y1

    mag = (dx * dx + dy * dy) ** 0.5
    dx /= mag
    dy /= mag

    lambd = (dx * (x3 - x1)) + (dy * (y3 - y1))
    x4 = (dx * lambd) + x1
    y4 = (dy * lambd) + y1

    res = VisiblePoint2D(x4, y4)
    return res


class AugmentedBodyParts(IntEnum):
    HEAD = 25
    NECK = 26
    CHEST = 27


class PoseDetector:

    def __init__(self, mode=False, upBody=True, smooth=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)

        self.results = []
        self.augmented_landmarks = []
        self.augmented_connections = []
        self.connections_list = []

    @staticmethod
    def _to_point(landmark):
        point_dict = {"x": landmark.x, "y": landmark.y}
        if landmark.HasField("visibility"):
            point_dict["visibility"] = landmark.visibility
        return VisiblePoint2D(**point_dict)

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
        half_proportion = (half_proportion.x / 4, half_proportion.y / 4)

        # See face proportions https://i.pinimg.com/originals/9d/e1/d1/9de1d16dab7a5d5a819bf351b21ac598.jpg
        # nose -> head top is 5.5 => 11*0.5
        head = VisiblePoint2D(
            x=(nose.x + 11 * half_proportion[0]),
            y=(nose.y + 11 * half_proportion[1]),
            visibility=nose.visibility
        )
        new_landmarks.append(head)

        # neck -> nose is 2.5 => 5*0.5
        neck = VisiblePoint2D(
            x=(nose.x - 5 * half_proportion[0]),
            y=(nose.y - 5 * half_proportion[1]),
            visibility=nose.visibility
        )
        new_landmarks.append(neck)

        # --- 27. Chest
        # chest == (LShoulder + RShoulder) / 2
        lshoulder = self._to_point(landmarks[PoseLandmark.LEFT_SHOULDER])
        rshoulder = self._to_point(landmarks[PoseLandmark.RIGHT_SHOULDER])
        chest = VisiblePoint2D(
            x=(rshoulder.x + lshoulder.x) / 2,
            y=(rshoulder.y + lshoulder.y) / 2,
            visibility=(rshoulder.visibility + lshoulder.visibility) / 2
        )
        new_landmarks.append(chest)

        return new_landmarks

    # See https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png
    def augment_connections(self, connections):
        return frozenset([
            # (AugmentedBodyParts.HEAD, AugmentedBodyParts.NECK),
            # (AugmentedBodyParts.NECK, PoseLandmark.LEFT_SHOULDER),
            # (AugmentedBodyParts.NECK, PoseLandmark.RIGHT_SHOULDER),
            # (AugmentedBodyParts.NECK, AugmentedBodyParts.CHEST),
            # (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.LEFT_SHOULDER),

            (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
            (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),

            (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
            (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),

            # (AugmentedBodyParts.CHEST, PoseLandmark.LEFT_HIP),
            # (AugmentedBodyParts.CHEST, PoseLandmark.RIGHT_HIP)
        ])

    def estimate_pose(self, source_image, draw=True):
        rgb_image = cv.cvtColor(source_image, cv.COLOR_BGR2RGB)

        self.results = self.pose.process(rgb_image)
        if self.results.pose_landmarks:
            self.augmented_landmarks = self.augment_landmarks(self.results.pose_landmarks.landmark)
            self.augmented_connections = self.augment_connections(self.mpPose.UPPER_BODY_POSE_CONNECTIONS)

            if draw:
                self.mpDraw.draw_landmarks(source_image, self.augmented_landmarks, self.augmented_connections)
        return source_image

    def get_landmarks(self) -> List[VisiblePoint2D]:
        return self.augmented_landmarks

    # def get_connections(self, img, draw=True):
    def get_connections(self, img):
        pixel_height, pixel_width, _ = img.shape

        landmarks: List[VisiblePoint2D] = self.get_landmarks()
        connections = self.augmented_connections
        debug = []
        if landmarks and connections:
            for (first_none_idx, second_node_idx) in connections:
                first_node, second_node = landmarks[first_none_idx], landmarks[second_node_idx]

                is_low_connection_visibility = (first_node.visibility < 0.6) or (second_node.visibility < 0.6)
                if is_low_connection_visibility:
                    self.connections_list.append(None)
                    self.connections_list.append(None)
                    continue

                debug.append((landmarks, first_node.visibility, second_node.visibility))
                first_node, second_node = (first_node.x, first_node.y), (second_node.x, second_node.y)

                first_node = (pixel_width * first_node[0], pixel_height * first_node[1])
                second_node = (pixel_width * second_node[0], pixel_height * second_node[1])

                self.connections_list.append(first_node)
                self.connections_list.append(second_node)

            # for id, lm in enumerate(self.augmented_landmarks):
            # for id, lm in enumerate(self.results.pose_landmarks.landmark):
            #     pixel_height, pixel_width, c = img.shape
            #     # print(id, lm)
            #     cx, cy = int(lm.x * pixel_width), int(lm.y * pixel_height)
            #     self.connections_list.append([id, cx, cy])
            #     if draw:
            #         cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
        return self.connections_list
