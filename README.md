# Real-Time Face Detection

A webcam face detector built with OpenCV's DNN module and the pretrained ResNet-10 SSD Caffe model. It draws face boxes and confidence scores, counts faces, displays FPS, and can process images and video files.

## Requirements

- Windows 10 or 11
- Python 3.10 or newer
- A webcam for live detection
- Internet access on first run to download the model files into `models/`

## Run from VS Code

1. Install Python 3.10+ and the Microsoft Python extension for VS Code.
2. In VS Code, choose **File → Open Folder...** and open this `RealTimeFaceDetection` folder.
3. Open **Terminal → New Terminal**.
4. Run `py -3 -m venv .venv`.
5. Activate the environment in PowerShell with ` .\.venv\Scripts\Activate.ps1`. If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat`, or use `run.bat`.
6. Run `python -m pip install -r requirements.txt`.
7. Select the `.venv` interpreter: press **Ctrl+Shift+P**, choose **Python: Select Interpreter**, then select the interpreter inside `.venv`.
8. Run `python src\main.py` in the terminal, or press **F5** and choose the Python debugger if prompted.
9. The model files download automatically the first time. Allow camera access when Windows asks. Press **Q** in the video window to quit.

You can skip the manual setup and run `run.bat` from File Explorer or the VS Code terminal. It creates the virtual environment, installs dependencies, and starts the application. Subsequent runs can use `run.bat` again.

## Usage

```powershell
python src\main.py                         # default camera (index 0)
python src\main.py --source 1              # second camera
python src\main.py --source sample.mp4     # video file
python src\main.py --source photo.jpg      # still image
python src\main.py --confidence 0.7        # stricter confidence threshold
python src\main.py --source sample.mp4 --output output\annotated.mp4
```

The source is a camera index, image path, or video path. Output recording supports webcam and video sources. Create the output directory first if it does not exist.

## Project structure

```text
RealTimeFaceDetection/
├── models/                 # downloaded model files (not committed)
├── src/
│   ├── camera.py           # camera/video capture
│   ├── detector.py         # model download, loading, and inference
│   ├── main.py             # command-line app
│   └── utils.py            # FPS and drawing helpers
├── .gitignore
├── README.md
├── requirements.txt
└── run.bat
```

The pretrained model files are downloaded from the official OpenCV GitHub repositories. If the download fails, check your internet connection and try again; no partial file will be treated as a valid model on the next run.

## Notes

- This detects faces; it does not identify who a person is.
- Performance depends on the computer, camera resolution, and number of faces.
- `cv2.imshow` needs a desktop session; this app is not intended for a headless server.
