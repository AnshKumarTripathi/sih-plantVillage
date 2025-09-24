#!/usr/bin/env python3
"""
Camera Test Script
Test camera availability and functionality
"""

import cv2
import sys

def test_cameras():
    """Test all available cameras."""
    print("🔍 Testing camera availability...")
    
    available_cameras = []
    
    # Test cameras 0-5
    for i in range(6):
        print(f"Testing camera {i}...")
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Camera {i}: Available and working")
                    print(f"   Frame shape: {frame.shape}")
                    available_cameras.append(i)
                else:
                    print(f"❌ Camera {i}: Opened but cannot read frames")
                cap.release()
            else:
                print(f"❌ Camera {i}: Cannot open")
        except Exception as e:
            print(f"❌ Camera {i}: Error - {str(e)}")
    
    print(f"\n📊 Summary:")
    if available_cameras:
        print(f"✅ Found {len(available_cameras)} working camera(s): {available_cameras}")
        return available_cameras
    else:
        print("❌ No working cameras found")
        print("\n🔧 Troubleshooting:")
        print("1. Check if camera is connected")
        print("2. Close other applications using camera (Zoom, Teams, etc.)")
        print("3. Check camera permissions")
        print("4. Try restarting your computer")
        return []

def test_camera_feed(camera_index=0):
    """Test camera feed for a specific camera."""
    print(f"\n📹 Testing camera feed for camera {camera_index}...")
    
    try:
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_index}")
            return False
        
        print("✅ Camera opened successfully")
        
        # Test reading frames
        for i in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Frame {i+1}: Shape {frame.shape}")
            else:
                print(f"❌ Frame {i+1}: Failed to read")
                cap.release()
                return False
        
        cap.release()
        print("✅ Camera feed test successful")
        return True
        
    except Exception as e:
        print(f"❌ Camera feed test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🌱 Plant Disease Detection - Camera Test")
    print("=" * 50)
    
    # Test all cameras
    available_cameras = test_cameras()
    
    # Test first available camera
    if available_cameras:
        test_camera_feed(available_cameras[0])
    else:
        print("\n❌ No cameras available for testing")
        sys.exit(1)
    
    print("\n✅ Camera test completed!")
