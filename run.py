from transformers import AutoTokenizer, AutoModel
import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
from ultralytics import YOLO
import cv2
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
import os
from RealtimeTTS import TextToAudioStream, SystemEngine

from config import *

import warnings
warnings.filterwarnings("ignore")

set_voice_lock()

print("- Initializing...")
USING_CUDA = torch.cuda.is_available()
print("- Using CUDA:", USING_CUDA)

parser = argparse.ArgumentParser()
parser.add_argument("-wm", "--whisper_model", default="base", help="Whisper voice model",
                    choices=["tiny", "base", "small", "medium", "large"])
parser.add_argument("-ym", "--yolo_model", default="8n",
                    help="YOLO model, 8n for yolov8n.pt.\ne.g. If you want to use yolov5s.pt, you can set it to \"5s\"")
parser.add_argument("-ie", "--input_english", action="store_false", default=True,
                    help="Bool: Is the input in English? If so, use a model which performs better on English. If not, use a model avialable for many languages, which is less accurate.")

# Translate: Automatically translate considering `input_english`
parser.add_argument("-t", "--translate", action='store_true', default=False,
                    help="Translate the transcription to \"English\" or \"Chinese\" based on the input language.")

# Camera
parser.add_argument("-co", "--camera_on", action='store_true', default=False,
                    help="Whether to use the camera to capture images. Note: YOLO object detection is not available if \"-co\" is False.")

# TTS
parser.add_argument("-tts", "--text_to_speech", action='store_true', default=False,
                    help="Whether to use the TTS engine.")
parser.add_argument("-te", "--TTS_engine", default="Zira",
                    help="Text to speech engine. Note: Not available if \"-tts\" is False.", type=str)

# Don't change the following settings
parser.add_argument("-eth", "--energy_threshold", default=1000,
                    help="Energy level for mic to detect. the smaller the value, the more sensitive the mic is, but the easier it is to detect noise.", type=int)
parser.add_argument("-rt", "--record_timeout", default=5,
                    help="How real time the recording is in seconds.", type=float)
parser.add_argument("-pt", "--phrase_timeout", default=2,
                    help="How much empty space between recordings before we consider it a new line in the transcription.", type=float)

args = parser.parse_args()

# Target language
if not args.input_english:
    target_language = "Chinese"
else:
    target_language = "English"

# --- 1. Transcription Settings (From https://github.com/davabase/whisper_real_time)
# The last time a recording was retrieved from the queue.
phrase_time = None
# We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
recorder = sr.Recognizer()
recorder.energy_threshold = args.energy_threshold
# Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
recorder.dynamic_energy_threshold = False

source = sr.Microphone(sample_rate=16000)

if args.translate:
    from translatepy import Translate
    TRANSLATOR = Translate()

record_timeout = args.record_timeout
phrase_timeout = args.phrase_timeout

# --- 2. Load / Download voice model
print("- Loading voice model...")
model_voice = args.whisper_model
if args.whisper_model != "large" and not args.input_english:
    model_voice = model_voice + ".en"
audio_model = whisper.load_model(model_voice, download_root=FOLDER_MODEL_VOICE)
print("- Voice model is ready: {model_voice} [1/3]")

# --- 3. Load / Download chat model
print("- Loading chat model...")
tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/chatglm3-6b", trust_remote_code=True, cache_dir=FOLDER_MODEL_CHAT)
model_chat = AutoModel.from_pretrained(
    "THUDM/chatglm3-6b", trust_remote_code=True, cache_dir=FOLDER_MODEL_CHAT).half().cuda()
model_chat = model_chat.eval()
print("- Chat model is ready: chatglm3-6b [2/3]")

is_camera_on = args.camera_on

def _set_role(is_input_english, is_camera_on):
    role = ROLE if is_input_english else ROLE_ZH
    if is_camera_on:
        if is_input_english:
            role += ROLE_EXTRA_CAMERA
        else:
            role += ROLE_EXTRA_CAMERA_ZH
    return role

def ask_ai(input_text: str, info_yolo, history=[]):
    if info_yolo:
        input_text += " " + info_yolo
    response, _history = model_chat.chat(tokenizer, input_text, history=history)
    return response, _history

# Initialize the AI with predefined role ---
role = _set_role(args.input_english, is_camera_on)
_, history = ask_ai(role, None)

# --- 4. Camera On/Off
if is_camera_on:
    print("- Camera is on!")
    if not os.path.exists(CAMERA_LOCK):
        print("- Note: Please use `python camera.py` to start the camera!!!")
        input("Press any key to continue...")

# --- 5. Load / Download YOLO model
if is_camera_on:
    print(f"- Loading YOLO model: yolov{args.yolo_model}.pt [3/3]")
    model_yolo = YOLO(os.path.join(FOLDER_MODEL_YOLO, f"yolov{args.yolo_model}.pt"))
    print("- YOLO model is ready! [3/3]")

# --- 6. TTS Engine
useTTS = args.text_to_speech
if useTTS:
    engine = SystemEngine(args.TTS_engine)
    stream = TextToAudioStream(engine)
    print("- TTS Engine is ready!")

print("- Everything is ready!")
input("Press any key to continue...")
remove_voice_lock()

# --- Create the data_queuw after the user getting ready
# Thread safe Queue for passing data from the threaded recording callback.
data_queue = Queue()

def record_callback(_, audio: sr.AudioData) -> None:
    """
    Threaded callback function to receive audio data when recordings finish.
    audio: An AudioData containing the recorded bytes.
    """
    # Grab the raw bytes and push it into the thread safe queue.
    data = audio.get_raw_data()
    data_queue.put(data)
    
# Create a background thread that will pass us raw audio bytes.
# We could do this manually but SpeechRecognizer provides a nice helper.
recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

transcription = ['']
answer = []
transcription_translated = ['']
answer_translated = []

def yolo_info(result):
    info = ""
    boxes = result.boxes.xywh.cpu()
    origin_shape = result.boxes.orig_shape
    clss = result.boxes.cls.int().cpu().tolist()
    names = result.names
    confs = result.boxes.conf.float().cpu().tolist()

    if len(clss) == 0:
        return None

    labels = []
    positions_start = []
    positions_end = []
    for box, cls, conf in zip(boxes, clss, confs):
        x, y, w, h = box
        labels.append(f"{names[cls]}: {conf:.2f}")
        positions_start.append(str(int(x - w / 2)) + ", " + str(int(y - h / 2)))
        positions_end.append(str(int(x + w / 2)) + ", " + str(int(y + h / 2)))
        
    info = "{IMAGE INFO: image_size=" + str(origin_shape) + ", "
    for i in range(len(labels)):
        info += "Object_" + str(i) + ": [" + labels[i].split(":")[0] + ", " + str(positions_start[i]) + " -> " + str(positions_end[i]) + "] "
    info += "}"
    print(info) # TODO: delete me
    return info

clear()
while True:
    try:
        now = datetime.utcnow()
        # Pull raw recorded audio from the queue.
        if not data_queue.empty() and not os.path.exists(VOICE_LOCK):
            phrase_complete = False
            set_voice_lock()
            print("- Voice detected!")
            # --- 0. YOLO: only when the camera is on
            if is_camera_on:
                # Set IMAGE TRIGGER
                set_image_trigger()

                # Load the last frame
                print("- Loading the last frame... (if not exists, a black frame will be used)")
                try:
                    last_frame = cv2.imread(PTH_LAST_FRAME)
                except:
                    last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Predict the frame
                print("- YOLO predicting...")
                results = model_yolo.predict(last_frame, verbose=False, device='cuda' if USING_CUDA else 'cpu')
                img = np.squeeze(results[0].plot()) # Convert to numpy array
                cv2.imwrite(PTH_LAST_FRAME_PREDICTED, img)
                info_yolo = yolo_info(results[0])

            # If enough time has passed between recordings, consider the phrase complete.
            # Clear the current working audio buffer to start over with the new data.
            if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                phrase_complete = True
            # This is the last time we received new audio data from the queue.
            phrase_time = now

            # Combine audio data from queue
            audio_data = b''.join(data_queue.queue)
            data_queue.queue.clear()

            # Convert in-ram buffer to something the model can use directly without needing a temp file.
            # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
            # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
            audio_np = np.frombuffer(
                audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # --- 1. Transcribe and translate the audio
            print("- Transcribing...")
            # Read the transcription.
            result = audio_model.transcribe(audio_np, fp16=USING_CUDA)
            text = result['text'].strip()
            if args.translate:
                print("- Translating...")
                try:
                    text_translated = TRANSLATOR.translate(
                        text, target_language).result
                except:
                    text_translated = "Error translating text..."

            # If we detected a pause between recordings, add a new item to our transcription.
            # Otherwise edit the existing one.
            if phrase_complete:
                transcription.append(text)
                if args.translate:
                    transcription_translated.append(text_translated)
            else:
                transcription[-1] = text
                if args.translate:
                    transcription_translated[-1] = text_translated

            # --- 2. Chat with the AI
            print("- AI is thinking...")
            respond, history = ask_ai(text, info_yolo, history=history)
            answer.append(respond)
            if args.translate:
                try:
                    respond_translated = TRANSLATOR.translate(
                        respond, "Chinese").result
                except:
                    respond_translated = "Error translating text..."
                answer_translated.append(respond_translated)

            # --- 3. Record the transcription to a buffer and log file
            # Write the transcription to a buffer and log file.
            write_into_buffer_input(transcription[-1], PATH_BUFFER_INPUT)
            if args.translate:
                write_into_log(transcription[-1], answer[-1], history, PTH_LOG, transcription_translated[-1], answer_translated[-1])
            else:
                write_into_log(transcription[-1], answer[-1], history, PTH_LOG)

            # --- 4. Clear the console to reprint the updated transcription.
            os.system('cls' if os.name == 'nt' else 'clear')
            if args.translate:
                for line_transcription, line_transcription_translated, line_answer, line_answer_translated in zip(transcription, transcription_translated, answer, answer_translated):
                    print(f"You: {line_transcription}")
                    print(f"You: {line_transcription_translated}")
                    print(f"AI:  {line_answer}")
                    print(f"AI:  {line_answer_translated}")
            else:
                for line_transcription, line_answer in zip(transcription, answer):
                    print(f"You: {line_transcription}")
                    print(f"AI:  {line_answer}")
            # Flush stdout.
            print('', flush=True)

            # --- 5. TTS
            stream.feed(respond)
            stream.play()

            # --- Tip: You can continue
            print("- You can continue...")

            remove_voice_lock()

            # Infinite loops are bad for processors, must sleep.
            sleep(0.25)
    except KeyboardInterrupt:
        break

clear()
print(f"- Your conversation has been saved to {PTH_LOG}!")
