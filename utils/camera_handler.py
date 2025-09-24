"""
Real-Time Camera Processing Utilities
Handles live camera feed processing for plant disease detection.
"""

import cv2
import numpy as np
import streamlit as st
import logging
import time
from typing import Optional, Dict, List, Callable
from queue import Queue
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraHandler:
    """Handles real-time camera processing with frame sampling optimization."""
    
    def __init__(self, frame_sample_rate: int = 5, max_fps: int = 30):
        self.camera = None
        self.is_running = False
        self.frame_sample_rate = frame_sample_rate
        self.max_fps = max_fps
        self.frame_count = 0
        self.processed_frames = 0
        self.last_prediction_time = 0
        self.performance_metrics = {
            "fps": 0,
            "processing_time": 0,
            "memory_usage": 0,
            "frame_drop_rate": 0
        }
        self.prediction_history = []
        self.max_history_size = 10
        
    def start_camera(self, camera_index: int = 0) -> bool:
        """Start camera capture."""
        try:
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                logger.error(f"Failed to open camera {camera_index}")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, self.max_fps)
            
            self.is_running = True
            logger.info(f"Camera {camera_index} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {str(e)}")
            return False
    
    def stop_camera(self):
        """Stop camera capture and release resources."""
        try:
            self.is_running = False
            
            if self.camera:
                self.camera.release()
                self.camera = None
            
            logger.info("Camera stopped and resources released")
            
        except Exception as e:
            logger.error(f"Error stopping camera: {str(e)}")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from camera."""
        try:
            if not self.camera or not self.is_running:
                return None
            
            ret, frame = self.camera.read()
            
            if not ret:
                logger.warning("Failed to capture frame")
                return None
            
            self.frame_count += 1
            return frame
            
        except Exception as e:
            logger.error(f"Error capturing frame: {str(e)}")
            return None
    
    def should_process_frame(self) -> bool:
        """Determine if current frame should be processed based on sampling rate."""
        return self.frame_count % self.frame_sample_rate == 0
    
    def process_frame_realtime(self, frame: np.ndarray, prediction_callback: Callable) -> Dict:
        """Process frame for real-time prediction."""
        try:
            start_time = time.time()
            
            # Check if we should process this frame
            if not self.should_process_frame():
                return {"processed": False, "reason": "Frame sampling"}
            
            # Preprocess frame for prediction
            processed_frame = self.preprocess_frame(frame)
            
            # Make prediction
            prediction_results = prediction_callback(processed_frame)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.update_performance_metrics(processing_time)
            
            # Store prediction in history
            self.add_to_prediction_history(prediction_results)
            
            self.processed_frames += 1
            self.last_prediction_time = time.time()
            
            logger.info(f"Frame processed in {processing_time:.3f}s")
            
            return {
                "processed": True,
                "prediction": prediction_results,
                "processing_time": processing_time,
                "frame_number": self.frame_count
            }
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return {"processed": False, "error": str(e)}
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for model input."""
        try:
            # Resize frame to model input size
            resized = cv2.resize(frame, (224, 224))
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values
            normalized = rgb_frame.astype(np.float32) / 255.0
            
            # Add batch dimension
            batch_frame = np.expand_dims(normalized, axis=0)
            
            return batch_frame
            
        except Exception as e:
            logger.error(f"Error preprocessing frame: {str(e)}")
            raise Exception(f"Frame preprocessing failed: {str(e)}")
    
    def update_performance_metrics(self, processing_time: float):
        """Update performance metrics."""
        try:
            current_time = time.time()
            
            # Calculate FPS
            if self.last_prediction_time > 0:
                time_diff = current_time - self.last_prediction_time
                if time_diff > 0:
                    self.performance_metrics["fps"] = 1.0 / time_diff
            
            # Update processing time
            self.performance_metrics["processing_time"] = processing_time
            
            # Calculate frame drop rate
            total_frames = self.frame_count
            processed_frames = self.processed_frames
            if total_frames > 0:
                self.performance_metrics["frame_drop_rate"] = (total_frames - processed_frames) / total_frames
            
            # Estimate memory usage (simplified)
            self.performance_metrics["memory_usage"] = self.estimate_memory_usage()
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    def estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        try:
            # Rough estimation based on frame size and history
            frame_memory = 640 * 480 * 3 * 4 / (1024 * 1024)  # 640x480 RGB float32
            history_memory = len(self.prediction_history) * 0.1  # 0.1MB per prediction
            return frame_memory + history_memory
            
        except Exception as e:
            logger.error(f"Error estimating memory usage: {str(e)}")
            return 0.0
    
    def add_to_prediction_history(self, prediction: Dict):
        """Add prediction to history with size limit."""
        try:
            prediction_with_timestamp = {
                **prediction,
                "timestamp": datetime.now().isoformat(),
                "frame_number": self.frame_count
            }
            
            self.prediction_history.append(prediction_with_timestamp)
            
            # Limit history size
            if len(self.prediction_history) > self.max_history_size:
                self.prediction_history.pop(0)
                
        except Exception as e:
            logger.error(f"Error adding to prediction history: {str(e)}")
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def get_prediction_history(self) -> List[Dict]:
        """Get prediction history."""
        return self.prediction_history.copy()
    
    def set_frame_sample_rate(self, rate: int):
        """Update frame sampling rate."""
        if rate > 0:
            self.frame_sample_rate = rate
            logger.info(f"Frame sample rate updated to {rate}")
    
    def get_camera_info(self) -> Dict:
        """Get camera information."""
        try:
            if not self.camera:
                return {"status": "Camera not initialized"}
            
            info = {
                "status": "Running" if self.is_running else "Stopped",
                "frame_width": int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "frame_height": int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": self.camera.get(cv2.CAP_PROP_FPS),
                "frame_count": self.frame_count,
                "processed_frames": self.processed_frames,
                "sample_rate": self.frame_sample_rate
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting camera info: {str(e)}")
            return {"status": "Error", "error": str(e)}

def create_camera_handler(frame_sample_rate: int = 5) -> CameraHandler:
    """Create and configure camera handler."""
    try:
        handler = CameraHandler(frame_sample_rate=frame_sample_rate)
        logger.info(f"Camera handler created with sample rate: {frame_sample_rate}")
        return handler
        
    except Exception as e:
        logger.error(f"Error creating camera handler: {str(e)}")
        raise Exception(f"Failed to create camera handler: {str(e)}")

@st.cache_data
def get_available_cameras() -> List[int]:
    """Get list of available camera devices."""
    available_cameras = []
    
    try:
        # Test cameras 0-5
        for i in range(6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        
        logger.info(f"Available cameras: {available_cameras}")
        return available_cameras
        
    except Exception as e:
        logger.error(f"Error detecting cameras: {str(e)}")
        return []

def save_prediction_log(predictions: List[Dict], filename: str = None) -> str:
    """Save prediction history to file."""
    try:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_log_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        logger.info(f"Prediction log saved to {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error saving prediction log: {str(e)}")
        raise Exception(f"Failed to save prediction log: {str(e)}")

def load_prediction_log(filename: str) -> List[Dict]:
    """Load prediction history from file."""
    try:
        with open(filename, 'r') as f:
            predictions = json.load(f)
        
        logger.info(f"Prediction log loaded from {filename}")
        return predictions
        
    except Exception as e:
        logger.error(f"Error loading prediction log: {str(e)}")
        raise Exception(f"Failed to load prediction log: {str(e)}")
