"""
Plant Disease Detection App
Main Streamlit application for detecting plant diseases using machine learning.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import time
import json
import os
from typing import Dict, List, Optional

# Import our utility modules
from utils.model_loader import load_model, get_model_info, verify_model_compatibility, get_disease_class_name
from utils.image_processor import preprocess_image_cached, load_image_from_bytes, validate_image, get_image_info
from utils.predictions import predict_disease_cached, format_prediction_results
from utils.camera_handler import create_camera_handler, get_available_cameras

# Configure page
st.set_page_config(
    page_title="🌱 Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        height: 20px;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcf7f);
        transition: width 0.3s ease;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .disease-info {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .treatment-info {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_disease_data():
    """Load disease information and treatment data."""
    try:
        # Load disease information
        disease_info_path = "assets/disease_info.json"
        treatment_info_path = "assets/treatment_data.json"
        
        disease_info = {}
        treatment_info = {}
        
        if os.path.exists(disease_info_path):
            with open(disease_info_path, 'r') as f:
                disease_info = json.load(f)
        
        if os.path.exists(treatment_info_path):
            with open(treatment_info_path, 'r') as f:
                treatment_info = json.load(f)
        
        return disease_info, treatment_info
        
    except Exception as e:
        st.error(f"Error loading disease data: {str(e)}")
        return {}, {}

def display_prediction_results(results: Dict):
    """Display prediction results in a user-friendly format."""
    if not results or "top_predictions" not in results:
        st.error("No prediction results available")
        return
    
    primary = results["top_predictions"][0]
    confidence = results["confidence_metrics"]
    
    # Main prediction card
    st.markdown(f"""
    <div class="prediction-card">
        <h3>🔍 Primary Prediction</h3>
        <h2>{primary['plant']} - {primary['disease']}</h2>
        <p><strong>Confidence:</strong> {primary['confidence_percentage']}%</p>
        <p><strong>Severity:</strong> {results['severity']}</p>
        <p><strong>Healthy:</strong> {'✅ Yes' if results['is_healthy'] else '❌ No'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence visualization
    st.markdown("### 📊 Confidence Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Confidence Level", confidence['confidence_level'])
    
    with col2:
        st.metric("Processing Time", f"{results['processing_time']:.3f}s")
    
    with col3:
        st.metric("Confidence Range", f"{confidence['confidence_range']:.3f}")
    
    # Confidence bar
    confidence_pct = primary['confidence_percentage']
    st.markdown(f"""
    <div class="confidence-bar">
        <div class="confidence-fill" style="width: {confidence_pct}%"></div>
    </div>
    <p style="text-align: center; margin-top: 0.5rem;"><strong>{confidence_pct}%</strong></p>
    """, unsafe_allow_html=True)
    
    # Alternative predictions
    if len(results["top_predictions"]) > 1:
        st.markdown("### 🔄 Alternative Predictions")
        alternatives = results["top_predictions"][1:6]  # Show top 5 alternatives
        
        for alt in alternatives:
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 0.5rem; margin: 0.5rem 0; border-radius: 5px;">
                <strong>{alt['plant']} - {alt['disease']}</strong> ({alt['confidence_percentage']}%)
            </div>
            """, unsafe_allow_html=True)

def display_disease_information(plant: str, disease: str, disease_info: Dict, treatment_info: Dict):
    """Display detailed disease information and treatment recommendations."""
    
    # Create a key for lookup
    disease_key = f"{plant}___{disease}"
    
    st.markdown("### ℹ️ Disease Information")
    
    if disease_key in disease_info.get("diseases", {}):
        info = disease_info["diseases"][disease_key]
        
        st.markdown(f"""
        <div class="disease-info">
            <h4>{info.get('name', disease)}</h4>
            <p><strong>Description:</strong> {info.get('description', 'No description available')}</p>
            <p><strong>Symptoms:</strong> {', '.join(info.get('symptoms', ['No symptoms listed']))}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"Detailed information for {plant} - {disease} is not available in our database.")
    
    # Treatment recommendations
    st.markdown("### 💊 Treatment Recommendations")
    
    if disease_key in treatment_info.get("treatments", {}):
        treatment = treatment_info["treatments"][disease_key]
        
        st.markdown(f"""
        <div class="treatment-info">
            <h4>Treatment Options</h4>
            <p><strong>Immediate Action:</strong> {treatment.get('immediate', 'No immediate action specified')}</p>
            <p><strong>Prevention:</strong> {treatment.get('prevention', 'No prevention tips available')}</p>
            <p><strong>Chemical Treatment:</strong> {treatment.get('chemical', 'No chemical treatment specified')}</p>
            <p><strong>Organic Treatment:</strong> {treatment.get('organic', 'No organic treatment specified')}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"Treatment recommendations for {plant} - {disease} are not available in our database.")

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🌱 Plant Disease Detection System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ Settings")
        
        # Model settings
        st.markdown("#### Model Configuration")
        model_path = st.text_input("Model Path", value="model/plant_disease_model.h5")
        
        # Frame sampling settings
        st.markdown("#### Camera Settings")
        frame_sample_rate = st.slider("Frame Sample Rate", min_value=1, max_value=20, value=5, 
                                    help="Process every Nth frame (higher = better performance)")
        
        # Confidence threshold
        confidence_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=1.0, 
                                       value=0.5, step=0.1, help="Minimum confidence for predictions")
        
        # Load disease data
        disease_info, treatment_info = load_disease_data()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📸 Image Upload", "📹 Live Camera", "🧠 Frame Predictions", "📊 Analysis", "ℹ️ About"])
    
    with tab1:
        st.markdown("### 📸 Upload Plant Image")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a plant leaf for disease detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width='stretch')
            
            # Image information
            with st.expander("📋 Image Information"):
                image_info = get_image_info(image)
                if image_info:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Dimensions", f"{image_info['shape'][1]}x{image_info['shape'][0]}")
                        st.metric("Channels", image_info['channels'])
                    with col2:
                        st.metric("Data Type", str(image_info['dtype']))
                        st.metric("Mean Value", f"{image_info['mean_value']:.2f}")
            
            # Process image button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Processing image..."):
                    try:
                        # Load model
                        model = load_model(model_path)
                        
                        # Verify model compatibility
                        if not verify_model_compatibility(model):
                            st.warning("Model compatibility issues detected. Results may be inaccurate.")
                        
                        # Preprocess image
                        image_bytes = uploaded_file.getvalue()
                        preprocessed_image = preprocess_image_cached(image_bytes)
                        
                        # Make prediction
                        results = predict_disease_cached(model, preprocessed_image)
                        
                        # Display results
                        display_prediction_results(results)
                        
                        # Display disease information
                        primary = results["top_predictions"][0]
                        display_disease_information(
                            primary['plant'], 
                            primary['disease'], 
                            disease_info, 
                            treatment_info
                        )
                        
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
    
    with tab2:
        st.markdown("### 📹 Live Camera Analysis")
        
        # Initialize camera handler in session state
        if 'camera_handler' not in st.session_state:
            st.session_state.camera_handler = None
        if 'camera_started' not in st.session_state:
            st.session_state.camera_started = False
        if 'last_prediction' not in st.session_state:
            st.session_state.last_prediction = None
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        if 'async_prediction_started' not in st.session_state:
            st.session_state.async_prediction_started = False
        if 'captured_frames' not in st.session_state:
            st.session_state.captured_frames = []
        if 'frame_predictions' not in st.session_state:
            st.session_state.frame_predictions = []
        
        # Camera controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎥 Start Camera", type="primary", disabled=st.session_state.camera_started):
                try:
                    # Create camera handler
                    st.session_state.camera_handler = create_camera_handler(frame_sample_rate)
                    
                    # Start camera
                    if st.session_state.camera_handler.start_camera():
                        st.session_state.camera_started = True
                        
                        # Clear previous captured frames
                        st.session_state.captured_frames = []
                        st.session_state.frame_predictions = []
                        
                        st.success("✅ Camera started successfully!")
                        st.rerun()  # Force rerun to start continuous display
                    else:
                        st.error("❌ Failed to start camera. Please check if camera is available and not being used by another application.")
                except Exception as e:
                    st.error(f"❌ Error starting camera: {str(e)}")
        
        with col2:
            if st.button("⏹️ Stop Camera", disabled=not st.session_state.camera_started):
                try:
                    # Process captured frames for predictions automatically BEFORE stopping camera
                    if st.session_state.captured_frames:
                        st.info(f"🔍 Processing {len(st.session_state.captured_frames)} captured frames...")
                        
                        # Process frames immediately
                        try:
                            # Load model
                            model = load_model(model_path)
                            
                            # Process each captured frame with progress
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, frame_data in enumerate(st.session_state.captured_frames):
                                status_text.text(f"Processing frame {i+1}/{len(st.session_state.captured_frames)}...")
                                
                                # Preprocess frame using the camera handler (still available)
                                processed_frame = st.session_state.camera_handler.preprocess_frame(frame_data['frame'])
                                
                                # Make prediction
                                prediction_result = predict_disease_cached(model, processed_frame)
                                
                                # Store prediction
                                st.session_state.frame_predictions.append({
                                    'frame_number': frame_data['frame_number'],
                                    'timestamp': frame_data['timestamp'],
                                    'prediction': prediction_result,
                                    'frame': frame_data['frame']
                                })
                                
                                # Update progress
                                progress_bar.progress((i + 1) / len(st.session_state.captured_frames))
                            
                            status_text.text("✅ Processing complete!")
                            st.success(f"✅ Processed {len(st.session_state.frame_predictions)} frames automatically!")
                            
                        except Exception as e:
                            st.error(f"❌ Auto-processing error: {str(e)}")
                    else:
                        st.warning("⚠️ No frames captured for prediction")
                    
                    # Now stop the camera
                    if st.session_state.camera_handler:
                        st.session_state.camera_handler.stop_camera()
                    st.session_state.camera_started = False
                    st.session_state.async_prediction_started = False
                    st.session_state.camera_handler = None
                    st.info("🛑 Camera stopped")
                    
                except Exception as e:
                    st.error(f"❌ Error stopping camera: {str(e)}")
        
        with col3:
            if st.button("🔄 Reset", disabled=not st.session_state.camera_started):
                st.session_state.last_prediction = None
                st.session_state.prediction_history = []
                st.success("🔄 Camera reset")
        
        # Camera status and live feed
        if st.session_state.camera_started and st.session_state.camera_handler:
            st.info("🟢 Camera is running")
            
            # Get camera info
            camera_info = st.session_state.camera_handler.get_camera_info()
            
            # Debug information
            with st.expander("🔍 Debug Information"):
                st.write(f"Camera Status: {camera_info.get('status', 'Unknown')}")
                st.write(f"Frame Count: {camera_info.get('frame_count', 0)}")
                st.write(f"Camera Running: {st.session_state.camera_handler.is_running}")
                st.write(f"Camera Object: {st.session_state.camera_handler.camera is not None}")
                
                # Manual test button
                if st.button("🧪 Test Camera Capture"):
                    try:
                        test_frame = st.session_state.camera_handler.capture_frame()
                        if test_frame is not None:
                            st.success("✅ Camera capture successful!")
                            st.write(f"Frame shape: {test_frame.shape}")
                            st.image(test_frame, caption="Test Frame", width=300)
                        else:
                            st.error("❌ Camera capture failed")
                    except Exception as e:
                        st.error(f"❌ Camera test error: {str(e)}")
            
            # Live camera feed
            st.markdown("### 📹 Live Camera Feed")
            
            # Create placeholder for camera feed
            camera_placeholder = st.empty()
            
            # Continuous camera feed display
            try:
                # Capture a new frame from camera
                frame = st.session_state.camera_handler.capture_frame()
                
                if frame is not None:
                    # Display current frame
                    camera_placeholder.image(frame, caption="Live Camera Feed", width='stretch')
                    st.success(f"✅ Camera feed active - Frame {st.session_state.camera_handler.frame_count}")
                    
                    # Store frames for later prediction (every 5th frame)
                    if st.session_state.camera_handler.should_process_frame():
                        import time
                        st.session_state.captured_frames.append({
                            'frame': frame.copy(),
                            'frame_number': st.session_state.camera_handler.frame_count,
                            'timestamp': time.time()
                        })
                        st.info(f"📸 Frame {st.session_state.camera_handler.frame_count} captured for prediction")
                    
                    # Auto-refresh for continuous feed
                    import time
                    time.sleep(0.1)
                    st.rerun()
                else:
                    camera_placeholder.warning("⚠️ No camera feed available")
                    st.error("❌ Failed to capture frame from camera")
                    
                    # Try to restart camera
                    st.warning("🔄 Attempting to restart camera...")
                    if st.button("🔄 Restart Camera"):
                        st.session_state.camera_handler.stop_camera()
                        if st.session_state.camera_handler.start_camera():
                            st.success("✅ Camera restarted successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to restart camera")
                    
            except Exception as e:
                st.error(f"❌ Camera error: {str(e)}")
                st.session_state.camera_started = False
            
            # Frame capture status
            st.markdown("#### 📸 Frame Capture Status")
            
            if st.session_state.captured_frames:
                st.success(f"✅ {len(st.session_state.captured_frames)} frames captured for prediction")
                
                # Show captured frames info
                with st.expander("📋 Captured Frames Details"):
                    for i, frame_data in enumerate(st.session_state.captured_frames):
                        st.write(f"Frame {i+1}: #{frame_data['frame_number']} - {frame_data['timestamp']:.2f}s")
            else:
                st.info("⏳ No frames captured yet (capturing every 5th frame)")
            
            # Automatic capture status
            st.info("📸 Frames are captured automatically every 5th frame while camera is running")
            
            # Performance metrics
            st.markdown("#### 📊 Performance Metrics")
            
            if st.session_state.camera_handler:
                metrics = st.session_state.camera_handler.get_performance_metrics()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("FPS", f"{metrics['fps']:.1f}")
                    st.metric("Processing Time", f"{metrics['processing_time']:.3f}s")
                with col2:
                    st.metric("Memory Usage", f"{metrics['memory_usage']:.1f} MB")
                    st.metric("Frame Drop Rate", f"{metrics['frame_drop_rate']:.1%}")
                with col3:
                    st.metric("Frame Count", camera_info.get('frame_count', 0))
                    st.metric("Sample Rate", f"Every {camera_info.get('sample_rate', 5)} frames")
            
            # Prediction history
            if st.session_state.prediction_history:
                with st.expander("📈 Prediction History"):
                    for i, pred in enumerate(reversed(st.session_state.prediction_history[-5:])):  # Show last 5
                        if pred.get("processed", False):
                            primary = pred["prediction"]["top_predictions"][0]
                            st.markdown(f"""
                            **{i+1}.** {primary['plant']} - {primary['disease']} 
                            ({primary['confidence_percentage']}%) - {pred['prediction']['severity']}
                            """)
            
            # Export predictions
            if st.session_state.prediction_history:
                if st.button("📥 Export Predictions"):
                    try:
                        from utils.camera_handler import save_prediction_log
                        filename = save_prediction_log(st.session_state.prediction_history)
                        st.success(f"✅ Predictions exported to {filename}")
                    except Exception as e:
                        st.error(f"❌ Export error: {str(e)}")
        
        else:
            st.info("🔴 Camera is not running")
            
            # Show available cameras
            with st.expander("📷 Available Cameras"):
                try:
                    available_cameras = get_available_cameras()
                    if available_cameras:
                        st.success(f"✅ Found {len(available_cameras)} camera(s): {available_cameras}")
                        st.info("💡 If camera detection fails, try:")
                        st.markdown("""
                        - Close other applications using the camera (Zoom, Teams, etc.)
                        - Check if camera permissions are granted
                        - Try restarting the application
                        - On Windows: Check Device Manager for camera issues
                        """)
                    else:
                        st.warning("⚠️ No cameras detected")
                        st.error("""
                        **Troubleshooting Steps:**
                        1. Check if camera is connected and working
                        2. Close other applications using the camera
                        3. Check camera permissions in system settings
                        4. Try restarting your computer
                        """)
                except Exception as e:
                    st.error(f"❌ Camera detection error: {str(e)}")
            
            # Instructions
            st.markdown("""
            ### 📋 How to Use Live Camera Analysis
            
            1. **Start Camera**: Click "🎥 Start Camera" to begin live feed
            2. **Enable Analysis**: Check "🔍 Enable Real-time Analysis" for predictions
            3. **Frame Sampling**: Camera processes every {frame_sample_rate} frames for performance
            4. **View Results**: See live predictions with confidence scores
            5. **Monitor Performance**: Check FPS and processing metrics
            6. **Export Data**: Save prediction history for analysis
            
            **💡 Tips:**
            - Ensure good lighting for better predictions
            - Keep the plant leaf in focus
            - Results update automatically as you move the camera
            - Higher frame sample rates = better performance, lower accuracy
            """.format(frame_sample_rate=frame_sample_rate))
    
    with tab3:
        st.markdown("### 🧠 Frame Predictions")
        
        if st.session_state.frame_predictions:
            st.success(f"✅ {len(st.session_state.frame_predictions)} frames already processed automatically!")
            
            # Show processing status
            st.info("🎉 Frames were processed automatically when camera was stopped")
            
            # Display predictions
            if st.session_state.frame_predictions:
                st.markdown("#### 📊 Prediction Results")
                
                for i, pred_data in enumerate(st.session_state.frame_predictions):
                    with st.expander(f"Frame {i+1} - #{pred_data['frame_number']}"):
                        # Display frame
                        st.image(pred_data['frame'], caption=f"Frame {pred_data['frame_number']}", width=300)
                        
                        # Display prediction
                        primary = pred_data['prediction']['top_predictions'][0]
                        confidence = pred_data['prediction']['confidence_metrics']
                        
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h4>🔍 Prediction Result</h4>
                            <h3>{primary['plant']} - {primary['disease']}</h3>
                            <p><strong>Confidence:</strong> {primary['confidence_percentage']}%</p>
                            <p><strong>Severity:</strong> {pred_data['prediction']['severity']}</p>
                            <p><strong>Healthy:</strong> {'✅ Yes' if pred_data['prediction']['is_healthy'] else '❌ No'}</p>
                            <p><strong>Processing Time:</strong> {pred_data['prediction']['processing_time']:.3f}s</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show disease information
                        display_disease_information(
                            primary['plant'], 
                            primary['disease'], 
                            disease_info, 
                            treatment_info
                        )
            
            # Clear predictions button
            if st.button("🗑️ Clear All Predictions"):
                st.session_state.frame_predictions = []
                st.session_state.captured_frames = []
                st.success("✅ All predictions cleared!")
                st.rerun()
                
        elif st.session_state.captured_frames:
            st.warning(f"📸 {len(st.session_state.captured_frames)} frames captured but not yet processed")
            st.info("💡 Stop the camera to automatically process all captured frames")
            
        else:
            st.info("📸 No frames captured yet. Go to 'Live Camera' tab to capture frames.")
            
            # Instructions
            st.markdown("""
            ### 📋 How to Use Frame Predictions
            
            1. **Start Camera**: Go to 'Live Camera' tab and start the camera
            2. **Capture Frames**: Camera automatically captures every 5th frame
            3. **Stop Camera**: Click 'Stop Camera' when done (frames processed automatically)
            4. **View Results**: See predictions for each captured frame
            
            **💡 Tips:**
            - Capture multiple frames for better analysis
            - Each frame is processed independently
            - Results are stored until you clear them
            - Processing happens automatically when you stop the camera
            """)

    with tab4:
        st.markdown("### 📊 Analysis Dashboard")
        
        # Model information
        with st.expander("🤖 Model Information"):
            try:
                model = load_model(model_path)
                model_info = get_model_info(model)
                
                if model_info:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Parameters", f"{model_info['total_params']:,}")
                    with col2:
                        st.metric("Layers", model_info['layers'])
                    with col3:
                        st.metric("Input Shape", str(model_info['input_shape']))
                
                # Model summary
                if st.button("Show Model Summary"):
                    summary = model.summary()
                    st.text(summary)
                    
            except Exception as e:
                st.error(f"Error loading model information: {str(e)}")
        
        # Disease classes
        with st.expander("🌿 Supported Disease Classes"):
            st.markdown("The model can detect diseases in the following plant categories:")
            
            plant_categories = {
                "🍎 Fruits": ["Apple", "Blueberry", "Cherry", "Grape", "Orange", "Peach", "Raspberry", "Strawberry"],
                "🥬 Vegetables": ["Corn (Maize)", "Pepper", "Potato", "Squash", "Tomato"],
                "🌾 Legumes": ["Soybean"]
            }
            
            for category, plants in plant_categories.items():
                st.markdown(f"**{category}**")
                st.markdown(", ".join(plants))
        
        # Performance metrics
        with st.expander("⚡ Performance Metrics"):
            st.markdown("""
            - **Model Size**: Lightweight CNN architecture
            - **Input Resolution**: 224x224 pixels
            - **Processing Time**: < 2 seconds per image
            - **Accuracy**: > 90% on test dataset
            - **Supported Formats**: JPG, PNG, JPEG
            """)
    
    with tab5:
        st.markdown("### ℹ️ About This Application")
        
        st.markdown("""
        ## 🌱 Plant Disease Detection System
        
        This application uses machine learning to detect plant diseases from leaf images.
        
        ### ✨ Features
        - **38 Disease Classes**: Supports multiple plant species and diseases
        - **Real-time Analysis**: Fast prediction with confidence scores
        - **Treatment Recommendations**: Get actionable advice for plant care
        - **Alternative Diagnoses**: See multiple prediction options
        - **Performance Monitoring**: Track processing metrics
        
        ### 🔬 Technology Stack
        - **Framework**: Streamlit
        - **ML Library**: TensorFlow/Keras
        - **Image Processing**: OpenCV, PIL
        - **Model**: Convolutional Neural Network (CNN)
        
        ### 📊 Model Architecture
        - **Input**: 224x224x3 RGB images
        - **Architecture**: Sequential CNN with 2 convolutional layers
        - **Output**: 38 disease classes with confidence scores
        - **Training**: PlantVillage dataset
        
        ### 🎯 How to Use
        1. **Upload Image**: Take a clear photo of a plant leaf
        2. **Analyze**: Click "Analyze Image" to get predictions
        3. **Review Results**: Check confidence scores and alternatives
        4. **Get Treatment**: Follow recommended treatment steps
        
        ### ⚠️ Important Notes
        - Ensure good lighting and clear focus
        - Include the entire leaf in the image
        - Results are for guidance only - consult experts for serious issues
        - Model works best with healthy, disease-free, or clearly diseased leaves
        """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; margin-top: 2rem;">
            <p>🌱 Plant Disease Detection System | Built with Streamlit & TensorFlow</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
