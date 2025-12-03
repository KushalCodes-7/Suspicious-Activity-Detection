# Violence Detection System

A web application that uses deep learning to detect violent content in videos. The system analyzes uploaded videos and classifies them as "Suspicious" or "Not Suspicious" based on the content.

## Features

- Upload videos for violence detection
- View analysis results with confidence scores
- Track prediction history
- Responsive design that works on desktop and mobile devices
- Drag and drop interface for easy file upload

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Clone this repository or download the source code
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure you have the trained model file (`movinet_violence_best.keras`) in the project root directory.
2. Run the Flask application:

```bash
python app.py
```

3. Open your web browser and navigate to:

```
http://127.0.0.1:5000/
```

## Usage

1. **Home Page**: View previous prediction history
2. **Detect Violence Page**: Upload a video for analysis
   - Click "Browse Files" or drag and drop a video file
   - Click "Analyze Video" to start the analysis
   - View the results which will indicate if the video is suspicious or not

## Project Structure

```
.
├── app.py                # Main application file
├── requirements.txt      # Python dependencies
├── movinet_violence_best.keras  # Trained model
├── static/               # Static files (CSS, JS, uploads)
│   ├── css/
│   │   └── style.css    # Custom styles
│   ├── js/
│   └── uploads/         # Directory for uploaded videos
├── templates/           # HTML templates
│   ├── base.html       # Base template
│   ├── index.html      # Home page
│   └── predict.html    # Prediction page
└── README.md           # This file
```

## Notes

- The application supports video files in MP4, AVI, MOV, and MKV formats
- Maximum file size is limited to 100MB
- For best results, use well-lit videos with clear visibility
- The system works best with videos that are between 5-30 seconds long

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
