# Microphone Distance Classification

Real-time audio classification system to detect microphone distance using machine learning.

## Autoinstall

```bash
git clone https://github.com/Lukysoon/mic_distance_experiment.git ; cd mic_distance_experiment ; chmod +x install.sh ; ./install.sh
```

## Prerequisites

Install required dependencies:

```bash
pip install numpy librosa pandas scipy scikit-learn xgboost matplotlib seaborn sounddevice
```

## Usage

### Step 1: Train the Model

First, organize your audio dataset in this structure:

```
dataset/
├── very_close/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── close/
│   ├── audio1.wav
│   └── ...
├── far/
│   ├── audio1.wav
│   └── ...
└── very_far/
    ├── audio1.wav
    └── ...
```

Then train the model:

(you must wait until the .pkl file is generated. It could take time when you have big dataset)

```bash
python3 AudioDistance.py
```

This will:
- Load audio files from the `dataset/` directory
- Extract features from audio chunks
- Train an XGBoost classifier
- Display feature importance
- Save the trained model to `trained_model.pkl`

### Step 2: Run Real-time Classification

Once you have a trained model, run the real-time classifier:

```bash
python3 realtime_classifier.py
```

Or specify a custom model path:

```bash
python3 realtime_classifier.py path/to/your/model.pkl
```

## Real-time Visualization

The real-time classifier will show 4 visualizations:

1. **Current Prediction** - Large display of current distance classification with confidence
2. **Prediction History** - Timeline showing classification over the last 20 seconds
3. **Class Probabilities** - Bar chart of current probabilities for each distance class
4. **Confidence Over Time** - Line graph showing how confident the model is over time

## Configuration

### AudioDistance.py

You can modify training parameters:

```python3
classifier = AudioDistanceClassifier(
    model_type='xgboost',  # or 'gradientboosting'
    n_classes=4            # number of distance categories
)
```

### realtime_classifier.py

You can adjust real-time parameters:

```python3
rt_classifier = RealtimeAudioClassifier(
    model_path='trained_model.pkl',
    sr=16000,              # sample rate
    chunk_duration=1.0,    # analyze every N seconds
    buffer_size=20         # keep last N predictions
)
```

## Troubleshooting

### "Model file not found" error
- Make sure you've run `AudioDistance.py` first to train and save a model

### Audio input issues
- Check your microphone permissions
- Verify your microphone is properly connected
- Try adjusting the sample rate if you get audio errors

### Memory issues
- Reduce `buffer_size` in realtime_classifier.py
- Reduce `chunk_duration` for faster processing

## Distance Classes

The default classes are:
- **very_close** - Less than 0.5 meters from microphone
- **close** - 0.5 to 1 meter
- **far** - 1 to 2 meters
- **very_far** - More than 2 meters

You can customize these by organizing your dataset accordingly.
