#!/usr/bin/env python
# coding: utf-8

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pyautogui

import utils
from pose_detector import PoseDetector

# It's not safe to use, but we had to to this for presentation :(
pyautogui.FAILSAFE = False

WINDOW_MAXIMAL_SIZE = 15
ACTION_TRIGGER_TIMEOUT = 5

CONFIGURATION_PATH = Path("./configuration.json")

VIDEOS_FOLDER = Path("./videos")
# SHORT_NEXT_4_INCORRECT = VIDEOS_FOLDER / "short_next_4incorrect"

SHORT_NEXT_1FORWARD = VIDEOS_FOLDER / "short_next_1forward"
SHORT_NEXT_1LEFT = VIDEOS_FOLDER / "short_next_2left"
BACK = VIDEOS_FOLDER / "slides-back"

_FRAME_PATTERN = "frame_*.jpg"
_DEBUG = True


@dataclass()
class Gesture:
    action: str
    sample_frame_window: list
    sample_connections_window: list


@dataclass
class PlotData:
    xs: list
    ys: list


def video_to_images(video_file: Path, out_dir: Path):
    """
    :param video_file: path to video file
    :param out_dir: path to output file
    :return: None
    """
    if not video_file.exists():
        raise ValueError(f"Video file does not exist: {video_file}")

    if not out_dir.exists():
        out_dir.mkdir()

    capture = cv.VideoCapture(str(video_file))

    count = 0
    while True:
        success, image = capture.read()
        if not success:
            break

        frame_out_path: Path = out_dir / f"frame_{count:0>3d}.jpg"
        cv.imwrite(frame_out_path, image)  # save frame as JPEG file
        count += 1


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
        return connections1_new, connections2_new, (0, 0)

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
    return connections1_new, connections2_new, (cosine_score, weighted_score)


def match_frames_window(
        sample_frame_window,
        real_frame_window,
        sample_connections_window,
        real_connections_window
) -> float:
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

    detector = PoseDetector()

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


def prepare_initial_real_frame(cap: cv.VideoCapture, detector: PoseDetector, window_length: int):
    real_frame_window = np.repeat(None, window_length)
    real_connections_window = []
    for i in range(window_length):
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


def infinity_worker(data_dict):
    cap = cv.VideoCapture(0)
    detector = PoseDetector()

    if _DEBUG:
        for video_name in data_dict:
            frames_dir_path = Path(video_name)
            video_path = frames_dir_path.parent / (frames_dir_path.name + ".mp4")
            print("Loading image:", frames_dir_path)
            video_to_images(video_path, frames_dir_path)

    sample_frames_lists = []
    for video_name in data_dict:
        frames_dir_path = Path(video_name)
        sample_frame_paths = list(sorted(frames_dir_path.glob(_FRAME_PATTERN)))
        window_size = min(WINDOW_MAXIMAL_SIZE, len(sample_frame_paths))

        sample_frame_window, sample_connections_window = initialize_sample(sample_frame_paths, window_size)
        gesture = Gesture(video_name, sample_frame_window, sample_connections_window)

        sample_frames_lists.append(gesture)

    real_frame_window, real_connections_window = prepare_initial_real_frame(cap, detector, WINDOW_MAXIMAL_SIZE)

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
        if len(real_frame_window) > 0:
            real_frame_window[-1] = frame

        real_connections_window[:-1] = real_connections_window[1:]
        real_connections_window[-1] = connections_list

        for gesture in sample_frames_lists:
            window_score = match_frames_window(gesture.sample_frame_window, real_frame_window,
                                               gesture.sample_connections_window, real_connections_window)
            data: PlotData = data_dict[gesture.action]
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
                if dist > 0.02 and avg_y > 0.8 and window_score > 0.95 and \
                        window_score > (global_max_y * 0.99) and window_score > min(avg_y * 1.04, max_y):
                    print(f"Window triggered with score: {window_score} for {gesture.action}. Max {global_max_y}, AVG: {avg_y}")

                    end_time = time.time()
                    action_time_diff = end_time - previous_action_time
                    if action_time_diff > ACTION_TRIGGER_TIMEOUT:
                        previous_action_time = time.time()
                        yield gesture.action

        cv.imshow("Image", frame)
        cv.waitKey(1)


def animate_plot(i, ax, data: PlotData):
    xs = data.xs[-505:]
    ys = data.ys[-505:]
    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('Actions')


def do_work(data_dict):
    for video_triggered in infinity_worker(data_dict):
        print("Action:", video_triggered)
        if video_triggered == "videos/short_next_1forward":
            pyautogui.press('space')
        elif video_triggered == "videos/slides-back":
            pyautogui.press('left')
        else:
            print("Nothing")


def main_without_plotting():
    config_file = CONFIGURATION_PATH

    data_dict = dict()
    video_to_action_dict = {}

    with open(config_file) as config_file:
        config = json.load(config_file)
        print(config)
        for entry in config:
            video = entry["src"].replace(".mp4", "")
            data_dict[video] = PlotData([], [])
            video_to_action_dict[video] = entry["action"]

    for video_triggered in infinity_worker(data_dict):
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


def main():
    config_file = CONFIGURATION_PATH

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
            an = animation.FuncAnimation(fig, animate_plot, fargs=(ax, data_dict[video]), interval=100)
            anims.append(an)

    threading.Thread(target=do_work, args=(data_dict,)).start()
    plt.show()


if __name__ == "__main__":
    main()
    # main_without_plotting()
