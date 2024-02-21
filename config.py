import os
import time
import json

ROLE = "You're an AI assistant, named \"Huwari\".\
        Your master is \"Sirius\", who is an undergraduate at XJTLU,\
        People will ask you questions, and you should answer them. "

ROLE_ZH = "你是一个AI助手，名叫\"Huwari\"。\
            你的主人是\"Sirius\"，他是一名在XJTLU大学的本科学生。\
            人们会问你问题，你需要回答他们。"

ROLE_EXTRA_CAMERA = "You're equiped with a camera so that you can see things.\n\
                    Note: You don't need to say the coordinates, instead, you should use words like \"left\", \"right\", \"up\", \"down\" to describe the position.\n\
                    The image information will be passed in a pair of brackets. For example, if you get \"{IMAGE INFO: image_size=(480, 640), Object_0: [person, 135, 192 -> 572, 480] }\".\
                    You should say \"There is a person at the bottom right of the image\"."

ROLE_EXTRA_CAMERA_ZH = "你装备有一个摄像头因此你能看到东西。\n\
                        注意：你不需要说出坐标，而是应该使用\"左\"，\"右\"，\"上\"，\"下\"等词来描述位置。\n\
                        图像信息将会以一对括号的形式传递。例如，如果你得到\"{IMAGE INFO: image_size=(480, 640), Object_0: [person, 135, 192 -> 572, 480] } \"。\
                        你应该说\"图像的右下方有一个人\"。"

CV_WINDOW_NAME = "YOLO WINDOW"

FOLDER_LOG = r"logs"
FOLDER_BUFFER = r"buffer"
FOLDER_CACHE = r".cache"
FOLDER_MODEL = os.path.join(FOLDER_CACHE, "models")  # "cache\models"
FOLDER_MODEL_VOICE = os.path.join(
    FOLDER_MODEL, "voice")  # "cache\models\voice"
FOLDER_MODEL_CHAT = os.path.join(FOLDER_MODEL, "chat")  # "cache\models\chat"
FOLDER_MODEL_YOLO = os.path.join(FOLDER_MODEL, "yolo")  # "cache\models\yolo"

PATH_BUFFER_INPUT = os.path.join(FOLDER_BUFFER, "input.txt")

CAMERA_LOCK = "camera.lock"
VOICE_LOCK = "recording.lock"
IMAGE_TRIGGER = "capturing.trigger"
PTH_LAST_FRAME = os.path.join(FOLDER_CACHE, "last_frame.jpg")
PTH_LAST_FRAME_PREDICTED = os.path.join(
    FOLDER_CACHE, "last_frame_predicted.jpg")

for folder in [FOLDER_LOG, FOLDER_BUFFER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

for txt in [PATH_BUFFER_INPUT]:
    if not os.path.exists(txt):
        with open(txt, "w") as f:
            f.write("")


TIME_NOW = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
FOLDER_LOG_NOW = os.path.join(FOLDER_LOG, TIME_NOW)
os.mkdir(FOLDER_LOG_NOW)
PTH_LOG = os.path.join(FOLDER_LOG_NOW, f"{TIME_NOW}.txt")
PTH_HISTORY = os.path.join(FOLDER_LOG_NOW, "history.json")
PTH_LOG_TRANSLATED = os.path.join(
    FOLDER_LOG_NOW, f"{TIME_NOW}_translated.txt")
if not os.path.exists(FOLDER_MODEL):
    os.makedirs(FOLDER_MODEL)


# --- Helper Functions ---
def write_into_buffer_input(text: str, path_buffer: str) -> None:
    with open(path_buffer, 'w', encoding='utf-8') as f:
        f.write(text)


def write_into_log(transcription: str, answer: str,  history, path_log: str, transcription_translated=None, answer_translated=None) -> None:
    with open(path_log, 'a', encoding='utf-8') as f:
        f.write(transcription + '\n')
        if transcription_translated:
            f.write(transcription_translated + '\n')
        f.write(answer + '\n')
        if answer_translated:
            f.write(answer_translated + '\n')
    # record the history to a json file
    with open(PTH_HISTORY, "w", encoding="utf-8") as f:
        json.dump(history, f)

def clear():
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")

# --- Setters and removers ---
def set_voice_lock():
    with open(VOICE_LOCK, "w") as f:
        f.write("")

def remove_voice_lock():
    if os.path.exists(VOICE_LOCK):
        os.remove(VOICE_LOCK)

def set_camera_lock():
    with open(CAMERA_LOCK, "w") as f:
        f.write("")

def remove_camera_lock():
    if os.path.exists(CAMERA_LOCK):
        os.remove(CAMERA_LOCK)

def set_image_trigger():
    with open(IMAGE_TRIGGER, "w") as f:
        f.write("")

def remove_image_trigger():
    if os.path.exists(IMAGE_TRIGGER):
        os.remove(IMAGE_TRIGGER)
