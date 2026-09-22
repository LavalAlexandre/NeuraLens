# NeuraLens

A histopathology acquisition assistant: clip a smartphone onto a microscope and NeuraLens tells you what you're looking at and whether the shot is usable. Built in one day at the Google "Solve for Healthcare & Life Sciences with Gemma" hackathon (08/07/2025).

![Poster](/assets/NeuraLens_poster.png)

## What's in the repo

| Path | What |
|------|------|
| `src/`, `notebooks/finetune.ipynb` | LoRA fine-tuning of MedGemma-4B (4-bit) to classify tissue type, zoom level and focus quality from microscope images |
| `app/` | Android app (Kotlin/Compose): live camera, captions from a Vertex AI endpoint, spoken with on-device text-to-speech |
| `decode_webservice/` | Small Sanic service that turns a COCO RLE segmentation mask into a contour overlay PNG |

## Results

On the Motic human-tissue dataset (556 images, 15 tissue types):

| Task | Accuracy before | F1 before | Accuracy after | F1 after |
|------|-----------------|-----------|----------------|----------|
| Tissue type | 0.422 | 0.333 | 0.679 | 0.648 |
| Zoom level | 0.211 | 0.148 | 0.404 | 0.377 |
| Focus quality | 0.101 | 0.021 | 0.862 | 0.824 |
| **Overall** | **0.245** | **0.167** | **0.648** | **0.616** |

These numbers come from the hackathon run, which used a random image-level split, so shots of the same slide could appear in both train and validation. The code now splits by slide, so expect lower (and more honest) numbers if you retrain.

The fine-tuned model is on Hugging Face: https://huggingface.co/MasterTrtle/Merge-medgemma-4b-it-lora-tissue-classifier

# Fine-tuning setup

Create a `.env` containing `HF_TOKEN`, put the images in `src/dataset/Motic-Human-tissues`, then:

```bash
conda create -n medgemma python=3.10
conda activate medgemma
pip install -r requirements.txt
```

Run `notebooks/finetune.ipynb` from the repo root. A CUDA GPU with bfloat16 support (Ampere or newer) is required.

# Downloading the app directly
https://drive.google.com/file/d/1IfqtY7GVd-CdcJSUv8jDuaYx3vD_pULk/view?usp=sharing 


# App Building

## Building and Debugging

This project uses Gradle for building. You can use Android Studio or the command line to build and debug the app.

### Prerequisites

- Java Development Kit (JDK) 17 or higher
- Android Studio (latest version recommended) or Android SDK command-line tools

### Building the APK

To build the debug APK from the command line, run the following command in the root directory of the project:

```bash
./gradlew assembleDebug
```

The generated APK will be located in `app/build/outputs/apk/debug/app-debug.apk`.

### Installing the APK

To install the APK on a connected Android device or emulator, use the Android Debug Bridge (adb):

```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

### Debugging

You can view the application logs for debugging purposes using `adb logcat`:

```bash
adb logcat
```

To filter the logs for this specific application, you can use a command like this:

```bash
adb logcat com.example.ollamacameraapp:V *:S
```

## Configuration

The app calls a Vertex AI endpoint (URL in `app/src/main/java/com/example/ollamacameraapp/network/VertexAiClient.kt`). Add your access token to `local.properties` (git-ignored) before building:

```properties
vertex.accessToken=ya29....
```

# Mask webservice

```bash
pip install -r decode_webservice/requirements.txt
python decode_webservice/decode_webservice.py   # listens on :8050
python decode_webservice/test_webservice.py     # sends a sample mask
```
