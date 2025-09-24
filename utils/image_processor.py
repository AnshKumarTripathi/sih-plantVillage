"""
Image Processing Utilities
Handles image preprocessing for plant disease detection model.
"""

import numpy as np
import cv2
from PIL import Image
import streamlit as st
import logging
from typing import Tuple, Optional, Union
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resize image to target size while maintaining aspect ratio.
    
    Args:
        image (np.ndarray): Input image
        target_size (Tuple[int, int]): Target size (width, height)
        
    Returns:
        np.ndarray: Resized image
    """
    try:
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        logger.info(f"Image resized to {target_size}")
        return resized
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        raise Exception(f"Failed to resize image: {str(e)}")

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to range [0, 1].
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        np.ndarray: Normalized image
    """
    try:
        # Convert to float and normalize
        normalized = image.astype(np.float32) / 255.0
        logger.info("Image normalized to [0, 1] range")
        return normalized
    except Exception as e:
        logger.error(f"Error normalizing image: {str(e)}")
        raise Exception(f"Failed to normalize image: {str(e)}")

def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert image to RGB format.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        np.ndarray: RGB image
    """
    try:
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 1:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:  # Already RGB
                pass
            else:
                raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
        else:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        logger.info(f"Image converted to RGB format: {image.shape}")
        return image
    except Exception as e:
        logger.error(f"Error converting to RGB: {str(e)}")
        raise Exception(f"Failed to convert image to RGB: {str(e)}")

def preprocess_image(image: Union[np.ndarray, Image.Image], target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Complete preprocessing pipeline for plant disease detection.
    
    Args:
        image (Union[np.ndarray, Image.Image]): Input image
        target_size (Tuple[int, int]): Target size for resizing
        
    Returns:
        np.ndarray: Preprocessed image ready for model input
    """
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is in correct format
        if image is None:
            raise ValueError("Input image is None")
        
        logger.info(f"Original image shape: {image.shape}")
        
        # Convert to RGB if needed
        image = convert_to_rgb(image)
        
        # Resize image
        image = resize_image(image, target_size)
        
        # Normalize pixel values
        image = normalize_image(image)
        
        # Add batch dimension for model input
        image = np.expand_dims(image, axis=0)
        
        logger.info(f"Preprocessed image shape: {image.shape}")
        return image
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        raise Exception(f"Image preprocessing failed: {str(e)}")

def validate_image(image: Union[np.ndarray, Image.Image]) -> bool:
    """
    Validate that the image is suitable for processing.
    
    Args:
        image (Union[np.ndarray, Image.Image]): Input image
        
    Returns:
        bool: True if image is valid, False otherwise
    """
    try:
        if image is None:
            logger.warning("Image is None")
            return False
        
        # Convert to numpy array for validation
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Check if image has valid dimensions
        if len(image.shape) < 2:
            logger.warning(f"Invalid image dimensions: {image.shape}")
            return False
        
        # Check minimum size
        if image.shape[0] < 32 or image.shape[1] < 32:
            logger.warning(f"Image too small: {image.shape}")
            return False
        
        # Check maximum size (to prevent memory issues)
        if image.shape[0] > 4096 or image.shape[1] > 4096:
            logger.warning(f"Image too large: {image.shape}")
            return False
        
        logger.info("Image validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating image: {str(e)}")
        return False

def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load image from bytes (useful for Streamlit file uploads).
    
    Args:
        image_bytes (bytes): Image data as bytes
        
    Returns:
        np.ndarray: Loaded image as numpy array
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        logger.info(f"Image loaded from bytes: {image_array.shape}")
        return image_array
        
    except Exception as e:
        logger.error(f"Error loading image from bytes: {str(e)}")
        raise Exception(f"Failed to load image from bytes: {str(e)}")

def get_image_info(image: Union[np.ndarray, Image.Image]) -> dict:
    """
    Get information about the image.
    
    Args:
        image (Union[np.ndarray, Image.Image]): Input image
        
    Returns:
        dict: Image information
    """
    try:
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        info = {
            "shape": image_array.shape,
            "dtype": image_array.dtype,
            "min_value": float(np.min(image_array)),
            "max_value": float(np.max(image_array)),
            "mean_value": float(np.mean(image_array)),
            "channels": image_array.shape[2] if len(image_array.shape) == 3 else 1
        }
        
        logger.info(f"Image info: {info}")
        return info
        
    except Exception as e:
        logger.error(f"Error getting image info: {str(e)}")
        return {}

@st.cache_data
def preprocess_image_cached(image_bytes: bytes, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Cached version of image preprocessing for Streamlit.
    
    Args:
        image_bytes (bytes): Image data as bytes
        target_size (Tuple[int, int]): Target size for resizing
        
    Returns:
        np.ndarray: Preprocessed image
    """
    try:
        # Load image from bytes
        image = load_image_from_bytes(image_bytes)
        
        # Validate image
        if not validate_image(image):
            raise ValueError("Invalid image")
        
        # Preprocess image
        processed_image = preprocess_image(image, target_size)
        
        logger.info("Image preprocessing completed (cached)")
        return processed_image
        
    except Exception as e:
        logger.error(f"Error in cached preprocessing: {str(e)}")
        raise Exception(f"Cached image preprocessing failed: {str(e)}")

def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """
    Enhance image quality for better prediction accuracy.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        np.ndarray: Enhanced image
    """
    try:
        # Convert to uint8 for OpenCV operations
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(image.shape) == 3:
            # Color image
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        logger.info("Image quality enhanced")
        return enhanced
        
    except Exception as e:
        logger.error(f"Error enhancing image quality: {str(e)}")
        # Return original image if enhancement fails
        return image
