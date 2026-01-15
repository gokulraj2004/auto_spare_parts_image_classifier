import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd
import plotly.graph_objects as go
import json
from pathlib import Path
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Auto Parts Classifier",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    h1 {
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
        font-size: 3em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2 {
        color: #2c3e50;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 10px;
    }
    
    h3 {
        color: #34495e;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 30px;
        border-radius: 5px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f1f3f5;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 10px 0;
    }
    
    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Class Name Dictionary
class_names = {
    0: 'AIR COMPRESSOR', 1: 'ALTERNATOR', 2: 'BATTERY', 3: 'BRAKE CALIPER',
    4: 'BRAKE PAD', 5: 'BRAKE ROTOR', 6: 'CAMSHAFT', 7: 'CARBERATOR', 8: 'COIL SPRING',
    9: 'CRANKSHAFT', 10: 'CYLINDER HEAD', 11: 'DISTRIBUTOR', 12: 'ENGINE BLOCK', 13: 'FUEL INJECTOR',
    14: 'FUSE BOX', 15: 'GAS CAP', 16: 'HEADLIGHTS', 17: 'IDLER ARM', 18: 'IGNITION COIL',
    19: 'LEAF SPRING', 20: 'LOWER CONTROL ARM', 21: 'MUFFLER', 22: 'OIL FILTER', 23: 'OIL PAN',
    24: 'OVERFLOW TANK', 25: 'OXYGEN SENSOR', 26: 'PISTON', 27: 'RADIATOR', 28: 'RADIATOR FAN',
    29: 'RADIATOR HOSE', 30: 'RIM', 31: 'SPARK PLUG', 32: 'STARTER', 33: 'TAILLIGHTS', 34: 'THERMOSTAT',
    35: 'TORQUE CONVERTER', 36: 'TRANSMISSION', 37: 'VACUUM BRAKE BOOSTER', 38: 'VALVE LIFTER',
    39: 'WATER PUMP'
}

# Load model once
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="models/compressed_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def get_model_details():
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = image.astype(np.float32)
    return image

def classify_image(image):
    interpreter = load_model()
    input_details, output_details = get_model_details()
    
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    class_idx = np.argmax(output_data[0])
    class_prob = output_data[0][class_idx] * 100
    class_name = class_names.get(class_idx, "Unknown Class")
    
    # Get top 5 predictions
    top_5_indices = np.argsort(output_data[0])[-5:][::-1]
    top_5_predictions = [(class_names[idx], output_data[0][idx] * 100) for idx in top_5_indices]
    
    return class_name, class_prob, output_data[0], top_5_predictions

def create_prediction_chart(top_5):
    df = pd.DataFrame(top_5, columns=['Part', 'Confidence'])
    
    fig = go.Figure(data=[
        go.Bar(
            y=df['Part'],
            x=df['Confidence'],
            orientation='h',
            marker=dict(
                color=df['Confidence'],
                colorscale='Viridis',
                showscale=False,
            ),
            text=df['Confidence'].round(2).astype(str) + '%',
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Top 5 Predictions',
        xaxis_title='Confidence Score (%)',
        yaxis_title='Auto Part',
        height=400,
        margin=dict(l=150),
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='rgba(255, 255, 255, 0)',
        font=dict(size=12),
    )
    
    return fig

# Sidebar
with st.sidebar:
    # Display hero image
    try:
        from pathlib import Path
        assets_path = Path("assets")
        # Try to find an image in assets folder
        image_files = list(assets_path.glob("*.jpeg")) + list(assets_path.glob("*.jpg")) + list(assets_path.glob("*.png"))
        if image_files:
            st.image(str(image_files[0]), use_column_width=True, caption="🚗 Automotive Parts")
            st.markdown("---")
    except:
        pass
    
    st.markdown("### 🎨 About This App")
    st.info(
        "🚗 **Auto Parts Image Classifier**\n\n"
        "A deep learning model trained to identify 40 different auto parts "
        "with 98.0% accuracy using MobileNetV2.\n\n"
        "**Features:**\n"
        "✅ Real-time classification\n"
        "✅ Top 5 predictions\n"
        "✅ Confidence scores\n"
        "✅ Batch processing\n"
        "✅ Interactive gallery"
    )
    
    st.markdown("### 📊 Model Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Classes", "40", "parts")
    with col2:
        st.metric("Accuracy", "98.0%", "MobileNetV2")
    
    st.markdown("### 📋 Available Classes")
    with st.expander("View all 40 classes", expanded=False):
        # Display classes vertically with numbers
        classes_list = list(class_names.values())
        classes_html = "<ol style='columns: 2; column-gap: 30px;'>"
        for idx, class_name in enumerate(classes_list, 1):
            classes_html += f"<li>{class_name}</li>"
        classes_html += "</ol>"
        st.markdown(classes_html, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        help="Show results only if confidence is above this threshold"
    )

# Main header
st.markdown("# 🚗 Auto Parts Image Classifier")
st.markdown("### *Identify car parts instantly with AI*")

# Display model info banner
col_banner1, col_banner2, col_banner3 = st.columns(3)
with col_banner1:
    st.metric("🧠 Primary Model", "MobileNetV2", "98.0% Accuracy")
with col_banner2:
    st.metric("📊 Classes", "40 Auto Parts", "Balanced Dataset")
with col_banner3:
    st.metric("⚡ Speed", "TFLite", "Optimized")

st.markdown("---")

# Tabs for different modes
tab1, tab2, tab3 = st.tabs(["📷 Single Image", "📁 Batch Processing", "📚 Gallery"])

with tab1:
    st.markdown("## Single Image Classification")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "bmp"],
            key="single_upload"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", width=600)
    
    with col2:
        st.markdown("### Prediction Results")
        
        if uploaded_file is not None:
            with st.spinner("🔍 Analyzing image..."):
                class_name, class_prob, all_probs, top_5 = classify_image(image)
            
            # Check confidence threshold
            if class_prob >= confidence_threshold:
                st.success(f"✅ Prediction successful!", icon="✅")
                
                # Main prediction card
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 25px; border-radius: 10px; color: white; text-align: center;">
                    <h1 style="color: white; margin: 0;">{class_name}</h1>
                    <h2 style="color: rgba(255,255,255,0.9); margin: 10px 0;">
                        {class_prob:.2f}% Confidence
                    </h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                st.markdown("### Key Metrics")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Predicted Class", class_name)
                
                with metric_col2:
                    st.metric("Confidence", f"{class_prob:.2f}%")
                
                with metric_col3:
                    top_class_name, top_conf = top_5[0]
                    st.metric("Top Match", top_class_name)
                
                # Top 5 predictions
                st.markdown("### Top 5 Predictions")
                fig = create_prediction_chart(top_5)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed table
                st.markdown("### Detailed Breakdown")
                df_predictions = pd.DataFrame(top_5, columns=['Auto Part', 'Confidence (%)'])
                df_predictions['Confidence (%)'] = df_predictions['Confidence (%)'].round(2)
                df_predictions['Rank'] = range(1, len(df_predictions) + 1)
                df_predictions = df_predictions[['Rank', 'Auto Part', 'Confidence (%)']]
                
                st.dataframe(df_predictions, use_container_width=True, hide_index=True)
                
            else:
                st.warning(
                    f"⚠️ Low confidence: {class_prob:.2f}% "
                    f"(below {confidence_threshold}% threshold)"
                )

with tab2:
    st.markdown("## Batch Processing")
    st.info("📁 Process multiple images from the input folder")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Instructions")
        st.markdown("""
        1. Add images to the `input/` folder
        2. Click 'Process Batch' button
        3. Results will be saved to `output/results.json`
        """)
    
    with col2:
        if st.button("🚀 Process Batch", use_container_width=True, key="process_batch"):
            st.info("Processing batch images from input folder...")
            
            input_path = Path("input")
            output_path = Path("output")
            output_path.mkdir(exist_ok=True)
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
            image_files = [f for f in input_path.iterdir() 
                          if f.is_file() and f.suffix.lower() in image_extensions]
            
            if not image_files:
                st.error("❌ No images found in the input folder")
            else:
                progress_bar = st.progress(0)
                results = []
                
                results_placeholder = st.empty()
                
                for idx, image_file in enumerate(image_files):
                    try:
                        image = Image.open(image_file).convert('RGB')
                        class_name, class_prob, all_probs, top_5 = classify_image(image)
                        
                        result = {
                            "filename": image_file.name,
                            "predicted_class": class_name,
                            "confidence": round(float(class_prob), 2),
                            "top_5_predictions": [{"class": name, "confidence": round(float(prob), 2)} 
                                                 for name, prob in top_5],
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        results.append(result)
                        
                    except Exception as e:
                        results.append({
                            "filename": image_file.name,
                            "error": str(e)
                        })
                    
                    progress = (idx + 1) / len(image_files)
                    progress_bar.progress(progress)
                
                # Save results
                output_file = output_path / "results.json"
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Display results summary
                st.success(f"✅ Batch processing complete! Processed {len(image_files)} images")
                
                # Results summary
                successful = len([r for r in results if "error" not in r])
                failed = len([r for r in results if "error" in r])
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                with summary_col1:
                    st.metric("Total Images", len(image_files))
                with summary_col2:
                    st.metric("Successful", successful)
                with summary_col3:
                    st.metric("Failed", failed)
                
                # Show results table
                st.markdown("### Results")
                results_data = []
                for r in results:
                    if "error" not in r:
                        results_data.append({
                            "Filename": r["filename"],
                            "Predicted Class": r["predicted_class"],
                            "Confidence": f"{r['confidence']}%"
                        })
                    else:
                        results_data.append({
                            "Filename": r["filename"],
                            "Predicted Class": "Error",
                            "Confidence": r["error"]
                        })
                
                df_results = pd.DataFrame(results_data)
                st.dataframe(df_results, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("## 🖼️ Image Gallery")
    
    input_path = Path("input")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = sorted([f for f in input_path.iterdir() 
                         if f.is_file() and f.suffix.lower() in image_extensions])
    
    if not image_files:
        st.warning("📁 No images found in the input folder")
    else:
        # Gallery header with stats
        col_header1, col_header2 = st.columns([3, 1])
        with col_header1:
            st.markdown(f"### 📸 Displaying {len(image_files)} image(s)")
        with col_header2:
            cols_per_row = st.select_slider(
                "Grid Size",
                options=[2, 3, 4, 5],
                value=3,
                key="gallery_grid"
            )
        
        st.markdown("---")
        
        # Create gallery with enhanced styling
        cols_per_row_actual = cols_per_row
        for i in range(0, len(image_files), cols_per_row_actual):
            cols = st.columns(cols_per_row_actual, gap="medium")
            
            for col_idx, col in enumerate(cols):
                if i + col_idx < len(image_files):
                    image_file = image_files[i + col_idx]
                    
                    with col:
                        # Image container with border
                        st.markdown(
                            f'<div style="border: 3px solid #667eea; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);">',
                            unsafe_allow_html=True
                        )
                        
                        image = Image.open(image_file).convert('RGB')
                        st.image(image, use_column_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Classify with progress
                        with st.spinner(f"🔍 Analyzing..."):
                            class_name, class_prob, _, top_5 = classify_image(image)
                        
                        # Result card with dynamic color based on confidence
                        confidence_color = "#2ecc71" if class_prob >= 80 else "#f39c12" if class_prob >= 60 else "#e74c3c"
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {confidence_color}99 0%, {confidence_color}dd 100%); 
                                    padding: 15px; border-radius: 8px; color: white; text-align: center; margin-top: 10px;">
                            <div style="font-size: 16px; font-weight: bold; margin-bottom: 8px;">{class_name}</div>
                            <div style="font-size: 24px; font-weight: bold;">{class_prob:.1f}%</div>
                            <div style="font-size: 12px; margin-top: 8px; opacity: 0.9;">Confidence Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # File name
                        st.caption(f"📄 {image_file.name}")
                        
                        # Top 3 predictions dropdown
                        with st.expander("🔝 Top Predictions"):
                            top_3_df = pd.DataFrame(top_5[:3], columns=['Part', 'Confidence (%)'])
                            top_3_df['Confidence (%)'] = top_3_df['Confidence (%)'].apply(lambda x: f"{x:.2f}%")
                            top_3_df['Rank'] = ['🥇', '🥈', '🥉']
                            top_3_df = top_3_df[['Rank', 'Part', 'Confidence (%)']]
                            st.dataframe(top_3_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Model Performance")
    st.markdown("**Training Set:** 6,917 images\n**Validation Set:** 200 images\n**Test Set:** 200 images\n**MobileNetV2:** 98.0%\n**EfficientNet:** 96.5%")

with col2:
    st.markdown("### 🏗️ Architecture")
    st.markdown("**Primary Model:** MobileNetV2\n**Fine-tuned Layers:** 3\n**Total Parameters:** ~3M\n**Format:** TensorFlow Lite (Optimized)\n**Alternative:** EfficientNet (96.5%)")

with col3:
    st.markdown("### 📚 Dataset")
    st.markdown("**Classes:** 40\n**Data Source:** Kaggle\n**Image Size:** 224×224\n**License:** Open Source")

st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "<small>🚗 Auto Parts Image Classifier | Built with Streamlit & TensorFlow</small>"
    "</div>",
    unsafe_allow_html=True
)
