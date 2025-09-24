"""
Model Loading and Caching Utilities
Handles TensorFlow model loading with Streamlit caching for optimal performance.
"""

import tensorflow as tf
import streamlit as st
import os
import logging
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_model(model_path: str = "model/plant_disease_model.h5") -> tf.keras.Model:
    """
    Load the plant disease detection model with caching.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        tf.keras.Model: Loaded TensorFlow model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        
        # Load the model
        model = tf.keras.models.load_model(model_path)
        
        # Verify model structure
        if model is None:
            raise Exception("Failed to load model - model is None")
        
        logger.info("Model loaded successfully")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise Exception(f"Failed to load model: {str(e)}")

def get_model_info(model: tf.keras.Model) -> dict:
    """
    Get model information and architecture details.
    
    Args:
        model (tf.keras.Model): Loaded TensorFlow model
        
    Returns:
        dict: Model information dictionary
    """
    try:
        model_info = {
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "total_params": model.count_params(),
            "layers": len(model.layers),
            "trainable_params": sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        }
        
        logger.info(f"Model info: {model_info}")
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return {}

def verify_model_compatibility(model: tf.keras.Model) -> bool:
    """
    Verify that the model is compatible with our expected input/output.
    
    Args:
        model (tf.keras.Model): Loaded TensorFlow model
        
    Returns:
        bool: True if model is compatible, False otherwise
    """
    try:
        # Check input shape (should be 224x224x3 for RGB images)
        expected_input = (None, 224, 224, 3)
        if model.input_shape != expected_input:
            logger.warning(f"Unexpected input shape: {model.input_shape}, expected: {expected_input}")
            return False
        
        # Check output shape (should have 38 classes)
        if len(model.output_shape) != 2 or model.output_shape[1] != 38:
            logger.warning(f"Unexpected output shape: {model.output_shape}, expected: (None, 38)")
            return False
        
        logger.info("Model compatibility verified")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying model compatibility: {str(e)}")
        return False

def get_model_summary(model: tf.keras.Model) -> str:
    """
    Get a string representation of the model summary.
    
    Args:
        model (tf.keras.Model): Loaded TensorFlow model
        
    Returns:
        str: Model summary as string
    """
    try:
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error getting model summary: {str(e)}")
        return "Error generating model summary"

# Disease class names mapping (38 classes from PlantVillage dataset)
DISEASE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot", 
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

def get_disease_class_name(class_index: int) -> str:
    """
    Get the disease class name from class index.
    
    Args:
        class_index (int): Class index (0-37)
        
    Returns:
        str: Disease class name
    """
    if 0 <= class_index < len(DISEASE_CLASSES):
        return DISEASE_CLASSES[class_index]
    else:
        return f"Unknown_Class_{class_index}"
