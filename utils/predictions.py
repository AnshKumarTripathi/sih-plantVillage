"""
Prediction Logic
Handles plant disease prediction using the trained model.
"""

import numpy as np
import tensorflow as tf
import streamlit as st
import logging
from typing import List, Dict, Tuple, Optional
import time
from .model_loader import get_disease_class_name, DISEASE_CLASSES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_disease(model: tf.keras.Model, preprocessed_image: np.ndarray) -> np.ndarray:
    """
    Make disease prediction using the trained model.
    
    Args:
        model (tf.keras.Model): Loaded TensorFlow model
        preprocessed_image (np.ndarray): Preprocessed image ready for model input
        
    Returns:
        np.ndarray: Prediction probabilities for all classes
    """
    try:
        start_time = time.time()
        
        # Make prediction
        predictions = model.predict(preprocessed_image, verbose=0)
        
        prediction_time = time.time() - start_time
        logger.info(f"Prediction completed in {prediction_time:.3f} seconds")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise Exception(f"Prediction failed: {str(e)}")

def get_top_predictions(predictions: np.ndarray, top_k: int = 5) -> List[Dict]:
    """
    Get top K predictions with confidence scores.
    
    Args:
        predictions (np.ndarray): Model predictions
        top_k (int): Number of top predictions to return
        
    Returns:
        List[Dict]: List of top predictions with class names and confidence scores
    """
    try:
        # Get top K indices
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        
        top_predictions = []
        for i, idx in enumerate(top_indices):
            class_name = get_disease_class_name(idx)
            confidence = float(predictions[0][idx])
            
            # Parse plant and disease from class name
            plant, disease = parse_class_name(class_name)
            
            prediction = {
                "rank": i + 1,
                "class_index": int(idx),
                "class_name": class_name,
                "plant": plant,
                "disease": disease,
                "confidence": confidence,
                "confidence_percentage": round(confidence * 100, 2)
            }
            
            top_predictions.append(prediction)
        
        logger.info(f"Top {top_k} predictions generated")
        return top_predictions
        
    except Exception as e:
        logger.error(f"Error getting top predictions: {str(e)}")
        return []

def parse_class_name(class_name: str) -> Tuple[str, str]:
    """
    Parse class name to extract plant and disease information.
    
    Args:
        class_name (str): Full class name (e.g., "Apple___Apple_scab")
        
    Returns:
        Tuple[str, str]: (plant, disease) tuple
    """
    try:
        if "___" in class_name:
            parts = class_name.split("___")
            plant = parts[0]
            disease = parts[1] if len(parts) > 1 else "Unknown"
        else:
            plant = "Unknown"
            disease = class_name
        
        return plant, disease
        
    except Exception as e:
        logger.error(f"Error parsing class name: {str(e)}")
        return "Unknown", "Unknown"

def get_prediction_confidence(predictions: np.ndarray) -> Dict:
    """
    Get detailed confidence analysis for predictions.
    
    Args:
        predictions (np.ndarray): Model predictions
        
    Returns:
        Dict: Confidence analysis
    """
    try:
        pred_array = predictions[0]
        
        # Get primary prediction
        primary_idx = np.argmax(pred_array)
        primary_confidence = float(pred_array[primary_idx])
        
        # Calculate confidence metrics
        confidence_metrics = {
            "primary_confidence": primary_confidence,
            "primary_percentage": round(primary_confidence * 100, 2),
            "max_confidence": float(np.max(pred_array)),
            "min_confidence": float(np.min(pred_array)),
            "mean_confidence": float(np.mean(pred_array)),
            "std_confidence": float(np.std(pred_array)),
            "confidence_range": float(np.max(pred_array) - np.min(pred_array))
        }
        
        # Determine confidence level
        if primary_confidence > 0.8:
            confidence_level = "Very High"
        elif primary_confidence > 0.6:
            confidence_level = "High"
        elif primary_confidence > 0.4:
            confidence_level = "Medium"
        elif primary_confidence > 0.2:
            confidence_level = "Low"
        else:
            confidence_level = "Very Low"
        
        confidence_metrics["confidence_level"] = confidence_level
        
        logger.info(f"Confidence analysis: {confidence_level} ({primary_confidence:.3f})")
        return confidence_metrics
        
    except Exception as e:
        logger.error(f"Error analyzing confidence: {str(e)}")
        return {}

def is_healthy_prediction(predictions: np.ndarray, threshold: float = 0.5) -> bool:
    """
    Determine if the prediction indicates a healthy plant.
    
    Args:
        predictions (np.ndarray): Model predictions
        threshold (float): Confidence threshold for healthy classification
        
    Returns:
        bool: True if plant appears healthy
    """
    try:
        # Get top prediction
        top_idx = np.argmax(predictions[0])
        top_class = get_disease_class_name(top_idx)
        top_confidence = float(predictions[0][top_idx])
        
        # Check if it's a healthy class and confidence is above threshold
        is_healthy = "healthy" in top_class.lower() and top_confidence > threshold
        
        logger.info(f"Healthy prediction: {is_healthy} (class: {top_class}, confidence: {top_confidence:.3f})")
        return is_healthy
        
    except Exception as e:
        logger.error(f"Error determining healthy prediction: {str(e)}")
        return False

def is_valid_plant_image(predictions: np.ndarray, confidence_threshold: float = 0.3) -> Dict:
    """
    Determine if the input image is likely a valid plant image.
    
    Args:
        predictions (np.ndarray): Model predictions
        confidence_threshold (float): Minimum confidence for valid prediction
        
    Returns:
        Dict: Validation results with confidence and recommendations
    """
    try:
        pred_array = predictions[0]
        
        # Get top prediction confidence
        top_confidence = float(np.max(pred_array))
        
        # Calculate confidence distribution
        confidence_std = float(np.std(pred_array))
        confidence_mean = float(np.mean(pred_array))
        
        # Check if prediction is reliable
        is_reliable = top_confidence > confidence_threshold
        
        # Check confidence distribution (low std = uncertain predictions)
        is_uncertain = confidence_std < 0.1  # Very low standard deviation indicates uncertainty
        
        # Determine if image is likely a plant
        is_likely_plant = is_reliable and not is_uncertain
        
        # Get recommendations
        if not is_likely_plant:
            if top_confidence < 0.2:
                recommendation = "Image may not contain a plant or is too unclear"
            elif is_uncertain:
                recommendation = "Image is unclear or not a plant leaf"
            else:
                recommendation = "Image quality may be too low for reliable prediction"
        else:
            recommendation = "Image appears to be a valid plant leaf"
        
        validation_result = {
            "is_valid": is_likely_plant,
            "confidence": top_confidence,
            "confidence_std": confidence_std,
            "confidence_mean": confidence_mean,
            "is_reliable": is_reliable,
            "is_uncertain": is_uncertain,
            "recommendation": recommendation,
            "threshold_used": confidence_threshold
        }
        
        logger.info(f"Image validation: {validation_result}")
        return validation_result
        
    except Exception as e:
        logger.error(f"Error validating plant image: {str(e)}")
        return {
            "is_valid": False,
            "confidence": 0.0,
            "recommendation": "Error validating image",
            "error": str(e)
        }

def get_prediction_quality_score(predictions: np.ndarray) -> Dict:
    """
    Calculate a quality score for the prediction based on confidence distribution.
    
    Args:
        predictions (np.ndarray): Model predictions
        
    Returns:
        Dict: Quality metrics and score
    """
    try:
        pred_array = predictions[0]
        
        # Calculate various quality metrics
        max_confidence = float(np.max(pred_array))
        mean_confidence = float(np.mean(pred_array))
        std_confidence = float(np.std(pred_array))
        
        # Calculate entropy (higher entropy = more uncertainty)
        entropy = -np.sum(pred_array * np.log(pred_array + 1e-10))
        
        # Calculate quality score (0-100)
        # Higher max confidence and lower entropy = better quality
        quality_score = (max_confidence * 100) - (entropy * 10)
        quality_score = max(0, min(100, quality_score))
        
        # Determine quality level
        if quality_score >= 80:
            quality_level = "Excellent"
        elif quality_score >= 60:
            quality_level = "Good"
        elif quality_score >= 40:
            quality_level = "Fair"
        elif quality_score >= 20:
            quality_level = "Poor"
        else:
            quality_level = "Very Poor"
        
        quality_metrics = {
            "quality_score": quality_score,
            "quality_level": quality_level,
            "max_confidence": max_confidence,
            "mean_confidence": mean_confidence,
            "std_confidence": std_confidence,
            "entropy": entropy,
            "is_high_quality": quality_score >= 60
        }
        
        logger.info(f"Prediction quality: {quality_metrics}")
        return quality_metrics
        
    except Exception as e:
        logger.error(f"Error calculating quality score: {str(e)}")
        return {
            "quality_score": 0,
            "quality_level": "Error",
            "error": str(e)
        }

def get_disease_severity(predictions: np.ndarray) -> str:
    """
    Determine disease severity based on prediction confidence.
    
    Args:
        predictions (np.ndarray): Model predictions
        
    Returns:
        str: Severity level
    """
    try:
        primary_confidence = float(np.max(predictions[0]))
        
        if primary_confidence > 0.9:
            return "Critical"
        elif primary_confidence > 0.7:
            return "High"
        elif primary_confidence > 0.5:
            return "Medium"
        elif primary_confidence > 0.3:
            return "Low"
        else:
            return "Uncertain"
            
    except Exception as e:
        logger.error(f"Error determining disease severity: {str(e)}")
        return "Unknown"

def get_alternative_diagnoses(predictions: np.ndarray, num_alternatives: int = 3) -> List[Dict]:
    """
    Get alternative disease diagnoses for comparison.
    
    Args:
        predictions (np.ndarray): Model predictions
        num_alternatives (int): Number of alternative diagnoses
        
    Returns:
        List[Dict]: List of alternative diagnoses
    """
    try:
        # Get top predictions excluding the primary one
        top_indices = np.argsort(predictions[0])[-(num_alternatives + 1):-1][::-1]
        
        alternatives = []
        for i, idx in enumerate(top_indices):
            class_name = get_disease_class_name(idx)
            confidence = float(predictions[0][idx])
            plant, disease = parse_class_name(class_name)
            
            alternative = {
                "rank": i + 2,  # Start from 2 since 1 is primary
                "class_name": class_name,
                "plant": plant,
                "disease": disease,
                "confidence": confidence,
                "confidence_percentage": round(confidence * 100, 2)
            }
            
            alternatives.append(alternative)
        
        logger.info(f"Generated {len(alternatives)} alternative diagnoses")
        return alternatives
        
    except Exception as e:
        logger.error(f"Error getting alternative diagnoses: {str(e)}")
        return []

@st.cache_data
def predict_disease_cached(model, preprocessed_image: np.ndarray) -> Dict:
    """
    Cached version of disease prediction for Streamlit.
    
    Args:
        model: Loaded TensorFlow model
        preprocessed_image (np.ndarray): Preprocessed image
        
    Returns:
        Dict: Complete prediction results
    """
    try:
        start_time = time.time()
        
        # Make prediction
        predictions = predict_disease(model, preprocessed_image)
        
        # Get top predictions
        top_predictions = get_top_predictions(predictions, top_k=5)
        
        # Get confidence analysis
        confidence_metrics = get_prediction_confidence(predictions)
        
        # Validate if image is likely a plant
        validation_result = is_valid_plant_image(predictions)
        
        # Get prediction quality score
        quality_metrics = get_prediction_quality_score(predictions)
        
        # Determine if healthy
        is_healthy = is_healthy_prediction(predictions)
        
        # Get disease severity
        severity = get_disease_severity(predictions)
        
        # Get alternative diagnoses
        alternatives = get_alternative_diagnoses(predictions, num_alternatives=3)
        
        total_time = time.time() - start_time
        
        results = {
            "predictions": predictions,
            "top_predictions": top_predictions,
            "confidence_metrics": confidence_metrics,
            "validation_result": validation_result,
            "quality_metrics": quality_metrics,
            "is_healthy": is_healthy,
            "severity": severity,
            "alternatives": alternatives,
            "processing_time": total_time,
            "timestamp": time.time()
        }
        
        logger.info(f"Complete prediction results generated in {total_time:.3f} seconds")
        return results
        
    except Exception as e:
        logger.error(f"Error in cached prediction: {str(e)}")
        raise Exception(f"Cached prediction failed: {str(e)}")

def format_prediction_results(results: Dict) -> str:
    """
    Format prediction results for display.
    
    Args:
        results (Dict): Prediction results
        
    Returns:
        str: Formatted results string
    """
    try:
        if not results or "top_predictions" not in results:
            return "No prediction results available"
        
        primary = results["top_predictions"][0]
        confidence = results["confidence_metrics"]
        
        formatted = f"""
**Primary Prediction:**
- Plant: {primary['plant']}
- Disease: {primary['disease']}
- Confidence: {primary['confidence_percentage']}%
- Severity: {results['severity']}
- Healthy: {'Yes' if results['is_healthy'] else 'No'}

**Confidence Analysis:**
- Level: {confidence['confidence_level']}
- Range: {confidence['confidence_range']:.3f}
- Processing Time: {results['processing_time']:.3f}s
        """
        
        return formatted.strip()
        
    except Exception as e:
        logger.error(f"Error formatting results: {str(e)}")
        return "Error formatting prediction results"
