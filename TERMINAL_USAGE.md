# 🚗 Auto Parts Image Classifier - Terminal Version

A simple terminal-based image classifier for identifying auto parts using TensorFlow Lite.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Input Images**:
   - Place your car part images in the `input/` folder
   - Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`

3. **Run the Classifier**:
   ```bash
   python app.py
   ```

## How It Works

The script will:
- Scan all images in the `input/` folder
- Classify each image using the TensorFlow Lite model
- Display predictions in the terminal (predicted class + top 5 predictions with confidence scores)
- Save results to `output/results.json`

## Output

Results are saved as JSON in `output/results.json`:
```json
[
  {
    "filename": "image.jpg",
    "predicted_class": "ALTERNATOR",
    "confidence": 95.42,
    "top_5_predictions": [
      {"class": "ALTERNATOR", "confidence": 95.42},
      {"class": "STARTER", "confidence": 3.21},
      ...
    ]
  }
]
```

## Model

The classifier uses a compressed TensorFlow Lite model (`models/compressed_model.tflite`) trained on 40 different auto parts classes with 93.5% test accuracy.

## Available Classes

AIR COMPRESSOR, ALTERNATOR, BATTERY, BRAKE CALIPER, BRAKE PAD, BRAKE ROTOR, CAMSHAFT, CARBERATOR, COIL SPRING, CRANKSHAFT, CYLINDER HEAD, DISTRIBUTOR, ENGINE BLOCK, FUEL INJECTOR, FUSE BOX, GAS CAP, HEADLIGHTS, IDLER ARM, IGNITION COIL, LEAF SPRING, LOWER CONTROL ARM, MUFFLER, OIL FILTER, OIL PAN, OVERFLOW TANK, OXYGEN SENSOR, PISTON, RADIATOR, RADIATOR FAN, RADIATOR HOSE, RIM, SPARK PLUG, STARTER, TAILLIGHTS, THERMOSTAT, TORQUE CONVERTER, TRANSMISSION, VACUUM BRAKE BOOSTER, VALVE LIFTER, WATER PUMP
