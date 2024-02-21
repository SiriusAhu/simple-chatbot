# Introduction
> A simple chatbot that can listen!

This is an easy-implemented chatbot (named `Huwari`) that can listen to your voice and give you a response. (Just for fun :))

This project is based on:
- [Whisper Real Time](https://github.com/davabase/whisper_real_time)
- [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
- [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS)

## Functions

- Real-time voice recognition
- [optional] Transcript translation
- [optional] YOLO object detection
- [optional] Real-time text-to-speech (Only English is supported)

> `argparse` is used to set the optional functions.

## What's good

- I manually chaged the cache path of all necessary models to the `.cache` folder, so that Windows users don't need to worry about the issue of Disk C.

# Usage
> For some unkown reason, `transformers` library seems to need to access `huggingface` model hub even though the model is already downloaded.
> So it's necessary to make sure you can access [`huggingface`](https://huggingface.co/). (You may try to set the proxy if the connection is not stable.)

## Simple Usage

If you just want to talk with the chatbot, you can simply run the following command:
```bash
python run.py
```

## [IMPORTANT] If YOLO object detection is needed

You should run the `camera.py` before running the `run.py` if you want to use YOLO object detection.
```bash
python camera.py
```

## Optional Functions

| Command                      | Default | Description                                                                                                          |
| ---------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------- |
| "-wm" or "--whisper_model"   | "base"  | Whisper voice model: ["tiny", "base", "small", "medium", "large"]                                                    |
| "-ym" or "--yolo_model"      | "8n"    | YOLO model, 8n for yolov8n.pt. e.g. If you want to use yolov5s.pt, you can set it to "5s"                            |
| "-ie" or "--input_english"   | False   | Bool: Is the input in English? This will affect: whisper model selection*, LLM role setting, transcript translation. |
| "-t" or "--translate"        | False   | Translate the transcription to \"English\" or \"Chinese\" based on the input language.                               |
| "-co" or "--camera_on"       | False   | Whether to use the camera to capture images. Note: YOLO object detection is not available if "-co" is False.         |
| "-tts" or "--text_to_speech" | False   | Whether to use the TTS engine.                                                                                       |
| "-te" or "--TTS_engine"      | "Zira"  | Text to speech engine. Note: Not available if "-tts" is False.                                                       |

- For Whisper models, 2 types are provided: "mixed" and "english". 
  The former one will automatically detect the input language.
  The latter one will only recognize the voice in English but with better performance.

## Examples

If you want to chat with yolo, you can run the following command:
```bash
python run.py -co
```

If you want to use YOLOv5s, use tts and translate (default: True) the transcription to English, you can run the following command:
```bash
python run.py -co -ym 5s -tts
```

## Not recommended to change the following settings

Here are some functions provided by the [Whisper Real Time](https://github.com/davabase/whisper_real_time) that I don't recommend to change:

| Command                        | Default | Description                                                                                                                 |
| ------------------------------ | ------- | --------------------------------------------------------------------------------------------------------------------------- |
| "-eth" or "--energy_threshold" | 1000    | Energy level for mic to detect. The smaller the value, the more sensitive the mic is, but the easier it is to detect noise. |
| "-rt" or "--record_timeout"    | 5       | How real time the recording is in seconds.                                                                                  |
| "-pt" or "--phrase_timeout"    | 2       | How much empty space between recordings before we consider it a new line in the transcription.                              |

# Installation

## Environment Setup

### [For Linux] Install `ffmpeg`

Repository [Whisper Real Time](https://github.com/davabase/whisper_real_time) requires `ffmpeg` to be installed for Linux users.

You may install `ffmpeg` by running the following command:

Debian:
```bash
sudo apt-get install ffmpeg
```

Arch:
```bash
sudo pacman -S ffmpeg
```

Fedora:
```bash
sudo dnf install ffmpeg
```

### Create your virtual environment
Create your virtual environment. `Python 3.9` is recommended. (My case: `Python 3.9.18`)

Conda:
```bash
conda create -n chatbot python=3.9.18
conda activate chatbot
```

### Install required dependencies
```bash
pip install -r requirements.txt
```
### Install `torch` dependencies
For some unkown reasons, `pip` always install `cpu` version through `requirements.txt`. So you may need to install `torch` dependencies manually.

#### For `CPU`:
```bash
pip install torch torchvision torchaudio
```

#### For `GPU`:
CUDA 11.8 as an example:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Install optional dependencies

After installing necessary dependencies, you may install optional dependencies depending on your needs.

YOLO object detection:
```bash
pip install -r requirements_yolo.txt
```

Real-time text-to-speech:
```bash
pip install -r requirements_tts.txt
```

# Something I'm not sure
1. I pass the role setting in different languages to the model, i.e. if input language is English, I pass the role setting in English to the model. But I'm not sure if it's necessary to do so. Maybe based on the model?

# TODO
- [ ] Organize the `argparse` part

# Based on
Thanks:
- [Whisper Real Time](https://github.com/davabase/whisper_real_time)
- [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
- [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS)