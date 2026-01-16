# 🚗 Auto Parts Image Classifier

![ezgif-18969da4836aa39a](https://github.com/user-attachments/assets/9bd2acdd-faf1-42fd-bc66-0a5897544518)

A powerful deep learning application for identifying 40 different automotive parts from images. Built with Streamlit, TensorFlow, and optimized for real-time predictions.

## 🌟 Features

### 📷 **Single Image Classification**
- Upload individual images (JPG, PNG, BMP)
- Get instant predictions with confidence scores
- View top 5 predictions with detailed breakdown
- Visual representation of prediction confidence

### 📁 **Batch Processing**
- Process multiple images from the `input/` folder simultaneously
- Automatic saving of results to `output/results.json`
- Progress tracking with success/failure metrics
- Detailed results table with confidence scores

### 📚 **Interactive Image Gallery**
- Browse and classify multiple images at once
- **Adjustable grid layout** (2, 3, 4, or 5 columns)
- **Color-coded confidence indicators**:
  - 🟢 Green: ≥80% confidence (High confidence)
  - 🟡 Orange: 60-80% confidence (Medium confidence)
  - 🔴 Red: <60% confidence (Low confidence)
- **Expandable top predictions** with medal rankings (🥇🥈🥉)
- Professional card-based UI with shadows and borders

### ⚙️ **Advanced Features**
- **Confidence threshold control** - Filter results based on confidence levels
- **Model performance metrics** - View accuracy and dataset statistics
- **Real-time analysis** with loading indicators
- **Responsive design** - Works on desktop and tablets

## 📊 Model Architecture

### Trained Models
This project uses transfer learning with two powerful CNN architectures:

| Model | Accuracy | Parameters | Format |
|-------|----------|------------|--------|
| **MobileNetV2** (Primary) | **98.0%** ✅ | ~3M | TFLite Optimized |
| **EfficientNet** | **96.5%** | ~10M | Keras (.h5) |

### Primary Model Specifications (MobileNetV2)
- **Base Model**: MobileNetV2 (Pre-trained on ImageNet)
- **Fine-tuned Layers**: 3 Dense layers
- **Input Size**: 224×224 pixels
- **Output Classes**: 40 auto parts
- **Test Accuracy**: 98.0%
- **Format**: TensorFlow Lite (Optimized for deployment)

## 🎯 Supported Auto Parts (40 Classes)

Air Compressor, Alternator, Battery, Brake Caliper, Brake Pad, Brake Rotor, Camshaft, Carburetor, Coil Spring, Crankshaft, Cylinder Head, Distributor, Engine Block, Fuel Injector, Fuse Box, Gas Cap, Headlights, Idler Arm, Ignition Coil, Leaf Spring, Lower Control Arm, Muffler, Oil Filter, Oil Pan, Overflow Tank, Oxygen Sensor, Piston, Radiator, Radiator Fan, Radiator Hose, Rim, Spark Plug, Starter, Taillights, Thermostat, Torque Converter, Transmission, Vacuum Brake Booster, Valve Lifter, Water Pump

## 📈 Dataset Information

- **Total Images**: 7,317 high-quality automotive part images
- **Training Set**: 6,917 images (balanced across all 40 classes)
- **Validation Set**: 200 images
- **Test Set**: 200 images
- **Image Resolution**: 224×224 pixels (RGB)
- **Model Accuracy**: 98.0% (MobileNetV2), 96.5% (EfficientNet)
- **Data Source**: [Kaggle - Car Parts 40 Classes](https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes)
- **License**: Open Source

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd auto_spare_parts_image_classifier
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## 📁 Project Structure

```
auto_spare_parts_image_classifier/
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── models/
│   ├── compressed_model.tflite   # TensorFlow Lite model (optimized)
│   └── EfficientNetB2.h5         # Full Keras model (reference)
├── notebooks/
│   ├── auto_parts_EDA.ipynb      # Exploratory Data Analysis
│   └── transfer-learning-*.ipynb # Model training notebook
├── Dataset/
│   ├── train/                    # Training images (40 subfolders)
│   ├── valid/                    # Validation images (40 subfolders)
│   └── test/                     # Test images (40 subfolders)
├── input/                        # Place images here for batch processing
├── output/                       # Batch processing results saved here
└── assets/                       # Supporting assets
```

## 🎮 How to Use

### **Single Image Classification**
1. Go to the "📷 Single Image" tab
2. Click "Choose an image" and select a photo of an auto part
3. View the prediction result with confidence score
4. See top 5 predictions in the breakdown table

### **Batch Processing**
1. Go to the "📁 Batch Processing" tab
2. Place multiple images in the `input/` folder
3. Click "🚀 Process Batch"
4. View results in the table
5. Find detailed results in `output/results.json`

### **Image Gallery**
1. Go to the "📚 Gallery" tab
2. Place images in the `input/` folder
3. Use the "Grid Size" slider to adjust layout (2-5 columns)
4. Each image shows:
   - Live classification
   - Color-coded confidence indicator
   - Expandable top 3 predictions
5. Confidence colors help you understand prediction reliability

## 🔧 Configuration

### Adjusting Confidence Threshold
Use the sidebar slider to filter results by confidence level. Only predictions above the threshold will be displayed.

### Adding New Images
- **For single classification**: Upload directly in the app
- **For batch processing**: Add images to the `input/` folder
- **Supported formats**: JPG, JPEG, PNG, BMP, GIF

## 📊 Performance Metrics

- **Inference Time**: ~100-200ms per image (CPU)
- **Model Size**: ~5MB (TFLite format, optimized)
- **Memory Usage**: Minimal (~300MB with all dependencies)
- **Accuracy**: 93.5% on test set

## 🔍 Understanding Results

### Confidence Score Interpretation
- **90-100%**: Highly confident prediction ✅
- **75-90%**: Good confidence 👍
- **60-75%**: Moderate confidence ⚠️
- **Below 60%**: Low confidence ❌

The model may struggle with:
- Poor image quality or blurry photos
- Extreme angles or unusual perspectives
- Similar-looking parts (e.g., different brake components)
- Partial or obscured parts

## � Training Details

### Model Development
- **Framework**: TensorFlow 2.x with Keras
- **Transfer Learning**: Pre-trained models fine-tuned on automotive parts
- **Primary Model**: MobileNetV2 (98.0% accuracy)
- **Alternative**: EfficientNet (96.5% accuracy)
- **Optimization**: Quantized to TFLite format for efficiency

### Performance Comparison
| Model | Accuracy | File Size | Speed | Training Notebook |
|-------|----------|-----------|-------|-------------------|
| **MobileNetV2** | **98.0%** ✅ | ~5MB | Very Fast | `transfer-learning-using-mobilenetv2-acc-99-08.ipynb` |
| **EfficientNet** | **96.5%** | ~15MB | Fast |  |

The MobileNetV2 model is deployed in the Streamlit app for optimal balance between accuracy and performance.

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| App won't start | Check Python version (3.8+) and reinstall requirements |
| Model loading error | Ensure `models/compressed_model.tflite` exists |
| Images not found in gallery | Place images in `input/` folder and refresh |
| Slow predictions | Close other applications; TFLite is already optimized |
| Out of memory | Reduce batch size or close other apps |

## 📝 Requirements

```
streamlit>=1.28.0
tensorflow>=2.12.0
numpy>=1.24.0
pandas>=1.5.0
Pillow>=9.5.0
plotly>=5.14.0
```

For detailed requirements, see `requirements.txt`

## 🤝 Contributing

This project can be improved by:
- Adding more training data
- Incorporating additional car part categories
- Improving model performance with newer architectures
- Adding multi-language support

## 📄 License

This project uses the Kaggle Car Parts dataset which is open source. Please refer to the [dataset source](https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes) for licensing details.

## 🙏 Acknowledgments

- **Dataset**: Kaggle - Car Parts 40 Classes
- **Framework**: TensorFlow, Keras, Streamlit
- **Transfer Learning Base**: MobileNetV2 (Google)

## 📧 Contact & Support

For issues, suggestions, or questions:
1. Check the troubleshooting section above
2. Review the model accuracy expectations
3. Ensure your images are clear and well-lit

---

**Built with ❤️ using Streamlit & TensorFlow**

*Last Updated: January 2026*
