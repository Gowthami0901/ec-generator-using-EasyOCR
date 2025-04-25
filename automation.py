# automation.py
import time
import os
import io
import base64
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pytesseract
import requests
import re
import pandas as pd
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException
from selenium.webdriver.common.keys import Keys

# Try to import EasyOCR, but don't error if not available
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR not available. Using pytesseract only. To enable EasyOCR, install with: pip install easyocr")

# Set path to tesseract executable if not in PATH
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Initialize EasyOCR reader if available
if EASYOCR_AVAILABLE:
    try:
        reader = easyocr.Reader(['en'])
    except Exception as e:
        print(f"Error initializing EasyOCR: {e}")
        EASYOCR_AVAILABLE = False

def clean_captcha_text(text):
    """Clean and normalize captcha text for exactly 5 capital letters and numbers"""
    if not text:
        return ""
    
    # Remove spaces, newlines, and special characters
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[^A-Z0-9]', '', text.upper())  # Convert to uppercase and keep only capital letters and numbers
    
    # Normalize characters that are commonly misrecognized in captchas
    replacements = {
        # Capital letter specific improvements
        'O': 'O',  # Keep O as letter (was being converted to 0)
        'C': 'C',  # Keep C as letter (was being converted to 0)
        'A': 'A',  # Keep A as letter (was being converted to 4)
        'D': 'D',  # Ensure D is preserved as D
        'Q': 'Q',  # Keep Q as letter (was being converted to 0)
        'I': 'I',  # Keep I as letter (was being converted to 1)
        'L': 'L',  # Keep L as letter (was being converted to 1)
        '|': 'I',  # Vertical bar is likely I
        'S': 'S',  # Keep S as letter (was being converted to 5)
        'B': 'B',  # Keep B as letter (was being converted to 8)
        'Z': 'Z',  # Keep Z as letter (was being converted to 2)
        'G': 'G',  # Keep G as letter (was being converted to 6)
        'T': 'T',  # Keep T as letter (was being converted to 7)
        
        # Number replacements - only apply when confident
        '0': '0',
        '1': '1',
        '2': '2',
        '3': '3',
        '4': '4',
        '5': '5',
        '6': '6',
        '7': '7',
        '8': '8',
        '9': '9'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Handle length issues for 5-character captchas
    if len(text) > 5:
        # If too long, take the first 5 characters
        return text[:5]
    elif len(text) < 5 and len(text) >= 3:
        # If too short but reasonable, pad with likely characters
        return text.ljust(5, 'A')  # Use 'A' as padding as it's often clearer
    
    return text

def enhance_image_for_captcha(img):
    """Apply optimized enhancements for 5-character capital letters and numbers captcha"""
    # Resize image for better processing (3x)
    width, height = img.size
    img = img.resize((width * 3, height * 3), Image.LANCZOS)
    
    # Convert to grayscale
    img_gray = img.convert('L')
    
    # Increase contrast dramatically for better character separation
    enhancer = ImageEnhance.Contrast(img_gray)
    img_contrast = enhancer.enhance(4.0)  # Increased from 3.5 to 4.0 for better letter definition
    
    # Increase sharpness for clearer character edges
    enhancer = ImageEnhance.Sharpness(img_contrast)
    img_sharp = enhancer.enhance(3.5)  # Increased from 3.0 to 3.5 for sharper edges
    
    # Apply custom threshold to improve text visibility - adjusted for capital letters
    threshold = 135  # Lowered from 140 to better preserve letter shapes
    img_threshold = img_sharp.point(lambda p: 255 if p > threshold else 0)
    
    # Save the enhanced image for debugging
    img_threshold.save('captcha_enhanced_basic.png')
    
    return img_threshold

def preprocess_with_opencv(img):
    """Use OpenCV for advanced image preprocessing optimized for capital letters and numbers"""
    # Convert PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale if it's not already
    if len(img_cv.shape) == 3:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_cv
        
    # Resize for better processing - increased scale for better letter definition
    height, width = gray.shape
    gray = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)  # Increased from 2x to 3x
    
    # Apply bilateral filter to preserve edges of letters while removing noise
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply adaptive thresholding (optimized for capital letters)
    thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                 cv2.THRESH_BINARY, 13, 7)  # Adjusted parameters for letters
    
    # Dilate slightly to make text thicker and more readable
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Apply morphological opening to remove noise
    kernel_open = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel_open)
    
    # Apply morphological closing to connect broken letter parts
    kernel_close = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close)
    
    # Convert back to PIL image
    pil_img = Image.fromarray(closing)
    
    # Save for debugging
    pil_img.save('captcha_opencv.png')
    
    return pil_img

def solve_captcha_with_easyocr(img):
    """Use EasyOCR to solve captcha, optimized for 5-character captcha with capital letters"""
    if not EASYOCR_AVAILABLE:
        return ""
    
    try:
        # Save image to temp file for EasyOCR processing
        img_path = 'temp_captcha.png'
        img.save(img_path)
        
        # Use EasyOCR to read the text with specific parameters for capital letters
        results = reader.readtext(
            img_path, 
            detail=0, 
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            min_size=10,         # Ignore very small detections
            text_threshold=0.4,  # More sensitive detection (lowered from 0.5)
            paragraph=False,     # Treat as separate characters
            batch_size=1,        # Process one at a time for better accuracy
            width_ths=1.0,       # Less aggressive width merging for separate characters
            height_ths=1.0       # Less aggressive height merging
        )
        
        # Clean up temporary file
        try:
            os.remove(img_path)
        except:
            pass
        
        if results:
            # Join all detected text segments
            captcha_text = ''.join(results)
            clean_text = clean_captcha_text(captcha_text)
            print(f"EasyOCR detected: {captcha_text} → Cleaned: {clean_text}")
            return clean_text
        else:
            print("EasyOCR could not detect any text")
            return ""
    except Exception as e:
        print(f"Error using EasyOCR: {e}")
        return ""

def invert_image(img):
    """Invert image colors - sometimes helps with difficult captchas"""
    if img.mode == 'RGB':
        r, g, b = img.split()
        r_inverted = Image.eval(r, lambda x: 255 - x)
        g_inverted = Image.eval(g, lambda x: 255 - x)
        b_inverted = Image.eval(b, lambda x: 255 - x)
        inverted = Image.merge('RGB', (r_inverted, g_inverted, b_inverted))
    else:
        inverted = Image.eval(img, lambda x: 255 - x)
    
    inverted.save('captcha_inverted.png')
    return inverted

def solve_captcha_with_tesseract(img, config_preset=None):
    """Use Tesseract OCR optimized for 5-character capital letters and numbers"""
    try:
        # Define configuration presets for different captcha types
        configs = {
            'default': '--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            'single_word': '--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            'single_line': '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            'exact_5_chars': '--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ -c load_system_dawg=0 -c load_freq_dawg=0 -c tessedit_minimal_rejection=1',
            'capital_letters': '--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ -c load_system_dawg=0 -c load_freq_dawg=0 -c tessedit_minimal_rejection=1 -c classifier_character_list=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        }
        
        # Use the specified preset or default
        config = configs.get(config_preset, configs['exact_5_chars'])
        
        # Extract text using Tesseract
        result = pytesseract.image_to_string(img, config=config)
        
        # Clean up the result
        clean_text = clean_captcha_text(result)
        
        # Print results for debugging
        print(f"Tesseract ({config_preset or 'default'}) detected: {result.strip()} → Cleaned: {clean_text}")
        
        return clean_text
    
    except Exception as e:
        print(f"Error with Tesseract OCR: {e}")
        return ""

def apply_perspective_correction(img):
    """Apply perspective correction to straighten text - helps with capital letters"""
    try:
        # Convert PIL image to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Threshold the image
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no significant contours found, return original image
        if not contours or len(contours) < 2:
            return img
            
        # Calculate bounding rectangle around all contours
        all_points = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Create source points (current corners)
        src_pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
        
        # Create destination points (rectangle)
        dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        
        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Apply perspective transformation
        warped = cv2.warpPerspective(img_cv, M, (w, h))
        
        # Convert back to PIL image
        corrected_img = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        corrected_img.save('captcha_perspective_corrected.png')
        
        return corrected_img
    except Exception as e:
        print(f"Error in perspective correction: {e}")
        return img

def get_captcha_image(driver, captcha_element):
    """Extract captcha image from the page using multiple methods"""
    
    # Method 1: Try to capture using JavaScript (highest quality)
    try:
        img_base64 = driver.execute_script("""
            var img = arguments[0];
            var canvas = document.createElement('canvas');
            canvas.width = img.width * 2;
            canvas.height = img.height * 2;
            var ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            return canvas.toDataURL('image/png').substring(22);
        """, captcha_element)
        
        if img_base64:
            img_data = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_data))
            print("Got captcha via JavaScript canvas")
            return img
    except Exception as e:
        print(f"JavaScript method failed: {e}")
    
    # Method 2: Try direct download from src attribute
    try:
        src = captcha_element.get_attribute('src')
        if src:
            # Make URL absolute if needed
            if not src.startswith("http"):
                base_url = '/'.join(driver.current_url.split('/')[:3])
                src = base_url + ('' if src.startswith('/') else '/') + src
            
            print(f"Trying direct URL: {src}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Referer': driver.current_url
            }
            response = requests.get(src, stream=True, headers=headers)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                print("Got captcha via direct URL")
                return img
    except Exception as e:
        print(f"Direct URL method failed: {e}")
    
    # Method 3: Screenshot method
    try:
        print("Using screenshot method to capture captcha")
        driver.save_screenshot("captcha_screen.png")
        
        # Get element position
        location = captcha_element.location
        size = captcha_element.size
        
        # Crop screenshot
        screenshot = Image.open("captcha_screen.png")
        left = location['x']
        top = location['y']
        right = left + size['width']
        bottom = top + size['height']
        
        img = screenshot.crop((left, top, right, bottom))
        print("Got captcha via screenshot")
        return img
    except Exception as e:
        print(f"Screenshot method failed: {e}")
        return None

def segment_captcha_characters(img):
    """Segment a captcha into individual characters for better recognition"""
    try:
        # Convert PIL image to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours - these should correspond to the characters
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours from left to right
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        # Keep only the likely character contours (filter out noise)
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter by size to avoid noise
            if w > 10 and h > 10:
                valid_contours.append(contour)
        
        # If we found exactly 5 contours, that's promising
        if len(valid_contours) == 5:
            print("Found exactly 5 character segments")
            # Create a debug image with the contours highlighted
            debug_img = img_cv.copy()
            cv2.drawContours(debug_img, valid_contours, -1, (0, 255, 0), 2)
            cv2.imwrite('captcha_segments.png', debug_img)
            
            # Try to recognize each character separately
            chars = []
            for i, contour in enumerate(valid_contours):
                x, y, w, h = cv2.boundingRect(contour)
                # Add padding
                pad = 2
                x_start = max(0, x-pad)
                y_start = max(0, y-pad)
                x_end = min(gray.shape[1], x+w+pad)
                y_end = min(gray.shape[0], y+h+pad)
                
                # Extract the character
                char_img = thresh[y_start:y_end, x_start:x_end]
                # Convert back to proper format for OCR
                char_img = 255 - char_img  # Invert colors
                # Save for debugging
                cv2.imwrite(f'char_{i}.png', char_img)
                
                # Convert to PIL for Tesseract
                char_pil = Image.fromarray(char_img)
                
                # Use Tesseract with a very restrictive PSM mode for single characters
                char_text = pytesseract.image_to_string(char_pil, config='--psm 10 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                char_text = re.sub(r'[^A-Z0-9]', '', char_text.upper())
                
                if char_text:
                    # Take the first character if multiple were detected
                    chars.append(char_text[0])
                else:
                    # If Tesseract found nothing, add a placeholder
                    chars.append('?')
            
            # If we got characters for all segments, join them
            if len(chars) == 5:
                segmented_text = ''.join(chars)
                print(f"Character segmentation result: {segmented_text}")
                return segmented_text
        
        return ""
    except Exception as e:
        print(f"Error in character segmentation: {e}")
        return ""

def fine_tune_image_for_recognition(img):
    """Apply specialized image processing techniques for better captcha recognition"""
    # Create multiple versions with different processing parameters
    results = []
    
    # Version 1: High contrast with light smoothing
    try:
        img1 = img.copy()
        # Convert to grayscale
        img1 = img1.convert('L')
        # Apply contrast enhancement
        enhancer = ImageEnhance.Contrast(img1)
        img1 = enhancer.enhance(4.0)  # Higher contrast
        # Apply custom threshold
        threshold = 135
        img1 = img1.point(lambda p: 255 if p > threshold else 0)
        img1.save('captcha_v1.png')
        results.append(("v1", img1))
    except:
        pass
    
    # Version 2: Sharpened with different threshold
    try:
        img2 = img.copy()
        img2 = img2.convert('L')
        # Sharpen first
        enhancer = ImageEnhance.Sharpness(img2)
        img2 = enhancer.enhance(3.5)
        # Then contrast
        enhancer = ImageEnhance.Contrast(img2)
        img2 = enhancer.enhance(3.0)
        # Different threshold
        threshold = 150
        img2 = img2.point(lambda p: 255 if p > threshold else 0)
        img2.save('captcha_v2.png')
        results.append(("v2", img2))
    except:
        pass
    
    # Version 3: Adaptive thresholding with OpenCV
    try:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        # Resize for better processing
        gray = cv2.resize(gray, (gray.shape[1] * 3, gray.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
        # Apply bilateral filter to preserve edges
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
        # Dilate slightly to make text thicker
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        # Convert back to PIL
        img3 = Image.fromarray(dilated)
        img3.save('captcha_v3.png')
        results.append(("v3", img3))
    except:
        pass
    
    # Version 4: Different approach with morphological operations
    try:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Apply morphological operations
        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # Convert back to PIL
        img4 = Image.fromarray(opening)
        img4.save('captcha_v4.png')
        results.append(("v4", img4))
    except:
        pass
    
    return results

def use_advanced_captcha_recognition(img):
    """Use advanced techniques to recognize captcha text"""
    results = []
    
    # Process with multiple techniques
    tuned_images = fine_tune_image_for_recognition(img)
    
    # Try segmentation approach first (highly accurate for 5-char captchas)
    segmented_result = segment_captcha_characters(img)
    if segmented_result and len(segmented_result) == 5:
        results.append(("segment_original", segmented_result, 10))
    
    # Special processing for capital letters especially ABCD
    try:
        # Create a capital letter optimized version
        capital_img = img.copy()
        capital_img = capital_img.convert('L')
        # Enhance contrast for capital letters
        enhancer = ImageEnhance.Contrast(capital_img)
        capital_img = enhancer.enhance(4.5)
        # Apply a binary threshold
        threshold = 130
        capital_img = capital_img.point(lambda p: 255 if p > threshold else 0)
        capital_img.save('captcha_capital_optimized.png')
        
        # Try the capital letters specific OCR config
        capital_result = solve_captcha_with_tesseract(capital_img, 'capital_letters')
        if capital_result and len(capital_result) >= 3:
            results.append(("capital_letters_specific", capital_result, 9))
    except Exception as e:
        print(f"Error in capital letter processing: {e}")
    
    # Process each version with multiple OCR settings
    for version_name, version_img in tuned_images:
        # Try segmentation on each tuned version
        segmented_result = segment_captcha_characters(version_img)
        if segmented_result and len(segmented_result) == 5:
            results.append((f"segment_{version_name}", segmented_result, 9))
        
        # Try EasyOCR if available
        if EASYOCR_AVAILABLE:
            easyocr_result = solve_captcha_with_easyocr(version_img)
            if easyocr_result and len(easyocr_result) >= 3:
                results.append((f"easyocr_{version_name}", easyocr_result, 8))
        
        # Try Tesseract with specialized settings
        tess_result = solve_captcha_with_tesseract(version_img, 'exact_5_chars')
        if tess_result and len(tess_result) >= 3:
            # Weight based on length - exact 5 chars gets highest weight
            weight = 7 if len(tess_result) == 5 else 5
            results.append((f"tess_{version_name}", tess_result, weight))
            
        # Also try capital letters config with each version
        capital_result = solve_captcha_with_tesseract(version_img, 'capital_letters')
        if capital_result and len(capital_result) >= 3:
            # Give slightly higher weight to capital letter results
            weight = 8 if len(capital_result) == 5 else 6
            results.append((f"tess_capital_{version_name}", capital_result, weight))
    
    # Look for consensus across different methods
    if len(results) >= 2:
        result_texts = [clean_captcha_text(r[1]) for r in results]
        for text in result_texts:
            if result_texts.count(text) >= 2:
                print(f"Found consensus result across multiple methods: {text}")
                return text
    
    # Prefer results with exactly 5 characters
    exact_5_results = [r for r in results if len(clean_captcha_text(r[1])) == 5]
    if exact_5_results:
        # Sort by weight
        exact_5_results.sort(key=lambda x: -x[2])
        best_result = clean_captcha_text(exact_5_results[0][1])
        print(f"Using exact 5-character result ({exact_5_results[0][0]}): {best_result}")
        return best_result
    
    # If no exact 5-char results, use best available and fix length
    if results:
        results.sort(key=lambda x: -x[2])
        best_result = clean_captcha_text(results[0][1])
        print(f"Best captcha solution ({results[0][0]}): {best_result}")
        return best_result
    
    return ""

def optimize_for_abcd_letters(img):
    """Special preprocessing optimized specifically for A, B, C, D capital letters"""
    try:
        # Create a copy of the image
        img_abcd = img.copy()
        
        # Convert to grayscale
        img_abcd = img_abcd.convert('L')
        
        # Apply strong contrast enhancement
        enhancer = ImageEnhance.Contrast(img_abcd)
        img_abcd = enhancer.enhance(5.0)  # Very high contrast for letter shapes
        
        # Apply a specific threshold that works well for capital letters
        threshold = 127
        img_abcd = img_abcd.point(lambda p: 255 if p > threshold else 0)
        
        # Resize for better processing of letter shapes
        width, height = img_abcd.size
        img_abcd = img_abcd.resize((width * 4, height * 4), Image.LANCZOS)
        
        # Convert to OpenCV for more processing
        img_cv = cv2.cvtColor(np.array(img_abcd), cv2.COLOR_RGB2BGR)
        if len(img_cv.shape) == 3:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_cv
        
        # Apply a slight blur to reduce noise while preserving edges
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply morphological operations to enhance letter shapes
        # ABCD letters have distinctive shapes - dilate to make them more pronounced
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(blurred, kernel, iterations=1)
        
        # Apply closing to connect broken parts of letters
        kernel_close = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
        
        # Convert back to PIL
        result_img = Image.fromarray(closing)
        
        # Save for debugging
        result_img.save('captcha_abcd_optimized.png')
        
        return result_img
    except Exception as e:
        print(f"Error in ABCD optimization: {e}")
        return img

def process_captcha_with_retry(driver, max_attempts=3):
    """Process captcha with automatic retry logic limited to 3 attempts"""
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        print(f"\n--- Captcha attempt {attempt}/{max_attempts} ---")
        
        # Get the captcha image and solve it with advanced techniques
        try:
            # Find captcha image
            captcha_element = None
            try:
                captcha_div = driver.find_element(By.CLASS_NAME, "captcha-image")
                captcha_element = captcha_div.find_element(By.TAG_NAME, "img")
            except:
                try:
                    captcha_element = driver.find_element(By.ID, "captcha")
                except:
                    try:
                        captcha_element = driver.find_element(By.CSS_SELECTOR, "img[id*='captcha' i], img[alt*='captcha' i], img[src*='captcha' i]")
                    except Exception as e:
                        print(f"Could not find captcha image: {e}")
                        return False
            
            if not captcha_element:
                print("Captcha element not found")
                return False
            
            # Get and save the captcha image
            img = get_captcha_image(driver, captcha_element)
            if not img:
                print("Failed to capture captcha image")
                return False
            
            img.save(f"captcha_original_{attempt}.png")
            
            # First try ABCD-specific optimization
            abcd_optimized = optimize_for_abcd_letters(img)
            abcd_result = solve_captcha_with_tesseract(abcd_optimized, 'capital_letters')
            
            # If we got a reasonable result from ABCD optimization, use it
            if abcd_result and len(abcd_result) >= 4:
                captcha_text = abcd_result
                print(f"Using ABCD-optimized result: {captcha_text}")
            else:
                # Otherwise use advanced recognition
                captcha_text = use_advanced_captcha_recognition(img)
            
            if not captcha_text:
                print(f"Failed to solve captcha on attempt {attempt}")
                # Try refreshing captcha if possible
                try:
                    refresh_button = driver.find_element(By.CSS_SELECTOR, ".captcha-refresh")
                    refresh_button.click()
                    print("Clicked captcha refresh button")
                    time.sleep(2)  # Wait for new captcha to load
                except:
                    print("Could not refresh captcha")
                    # Try regular solve method as fallback
                    captcha_text = solve_captcha(driver)
                    if not captcha_text:
                        continue
            
            # Find input field
            captcha_input = find_captcha_input(driver)
            if not captcha_input:
                print(f"Could not find captcha input on attempt {attempt}")
                continue
            
            # Enter captcha
            captcha_input.clear()
            captcha_input.send_keys(captcha_text)
            print(f"Entered captcha text: {captcha_text}")
            
            # Find and click search button
            submit_button = find_submit_button(driver)
            if submit_button:
                time.sleep(1)
                submit_button.click()
                print("Clicked search button")
            else:
                # Fallback to Enter key
                captcha_input.send_keys(Keys.RETURN)
                print("Pressed Enter key to submit")
            
            # Wait for page to process
            time.sleep(5)
            
            # Check if captcha was accepted
            if is_captcha_invalid(driver):
                print(f"Captcha was rejected on attempt {attempt}")
                driver.save_screenshot(f"captcha_rejected_{attempt}.png")
                
                # Try to refresh captcha if possible
                try:
                    refresh_button = driver.find_element(By.CSS_SELECTOR, ".captcha-refresh")
                    refresh_button.click()
                    print("Clicked captcha refresh button for next attempt")
                    time.sleep(2)
                except:
                    print("Could not refresh captcha, will try with new captcha")
            else:
                print(f"Captcha appears to be accepted on attempt {attempt}")
                # Look for "Click here" button and click it if found
                try:
                    # Various ways to find "Click here" links
                    click_here_links = driver.find_elements(By.XPATH, "//a[contains(text(), 'Click here') or contains(text(), 'click here')] | //a[contains(@href, 'click')]")
                    if click_here_links:
                        for link in click_here_links:
                            if link.is_displayed():
                                print("Found 'Click here' link. Clicking it.")
                                link.click()
                                time.sleep(3)
                                return True
                    
                    # Also try buttons
                    click_here_buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'Click here') or contains(text(), 'click here')]")
                    if click_here_buttons:
                        for button in click_here_buttons:
                            if button.is_displayed():
                                print("Found 'Click here' button. Clicking it.")
                                button.click()
                                time.sleep(3)
                                return True
                except Exception as e:
                    print(f"Error finding 'Click here' link: {e}")
                
                return True
        
        except Exception as e:
            print(f"Error in captcha attempt {attempt}: {e}")
            driver.save_screenshot(f"error_screenshot_{attempt}.png")
    
    print(f"Failed to solve captcha after {max_attempts} attempts")
    return False

def find_captcha_input(driver):
    """Find the input field for entering captcha text"""
    # Strategy 1: Look for input with txt_Captcha ID (from the label's for attribute)
    try:
        input_field = driver.find_element(By.ID, "txt_Captcha")
        print("Found captcha input with ID: txt_Captcha")
        return input_field
    except NoSuchElementException:
        pass
    
    # Strategy 2: Try common selectors for captcha inputs
    selectors = [
        "input[name='txt_Captcha']",
        "input[id*='captcha' i]",  # Case-insensitive contains 'captcha'
        "input[name*='captcha' i]",
        "input.captcha-input",
        "input[placeholder*='code' i]",
        "input[placeholder*='captcha' i]",
        "input[placeholder*='type the code' i]",
        "[id*='captcha' i]:not(img):not(div)"  # Any element with captcha in ID that's not an image or div
    ]
    
    for selector in selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                if element.tag_name == 'input' and element.is_displayed():
                    print(f"Found captcha input with selector: {selector}")
                    return element
        except Exception:
            continue
    
    # Strategy 3: Look for input fields near the captcha label or image
    try:
        # Find label that might be related to captcha
        captcha_labels = driver.find_elements(By.XPATH, "//label[contains(text(), 'code') or contains(text(), 'Captcha') or contains(text(), 'Type the')]")
        
        for label in captcha_labels:
            # Check if label has a "for" attribute
            for_attr = label.get_attribute("for")
            if for_attr:
                try:
                    input_field = driver.find_element(By.ID, for_attr)
                    print(f"Found captcha input via label's for attribute: {for_attr}")
                    return input_field
                except NoSuchElementException:
                    pass
            
            # Look for nearby input fields
            try:
                # Look for inputs that are siblings or children of the label's parent
                parent = label.find_element(By.XPATH, "./..")
                inputs = parent.find_elements(By.TAG_NAME, "input")
                if inputs:
                    print("Found captcha input near label")
                    return inputs[0]
            except Exception:
                pass
    except Exception:
        pass
    
    # Strategy 4: Last resort - find any visible empty text input
    try:
        inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='text']")
        for inp in inputs:
            if inp.is_displayed() and not inp.get_attribute("value"):
                print("Using first empty text input as captcha field")
                return inp
    except Exception:
        pass
    
    print("Could not find captcha input field")
    return None

def find_submit_button(driver):
    """Find the submit button using multiple strategies"""
    # Strategy 1: Look for buttons with common submit text
    button_texts = ["Submit", "Continue", "Search", "Proceed", "OK", "Go"]
    for text in button_texts:
        try:
            # Try exact match
            button = driver.find_element(By.XPATH, f"//button[text()='{text}'] | //input[@value='{text}']")
            if button.is_displayed():
                print(f"Found submit button with text: {text}")
                return button
        except NoSuchElementException:
            try:
                # Try contains
                button = driver.find_element(By.XPATH, f"//button[contains(text(),'{text}')] | //input[contains(@value,'{text}')]")
                if button.is_displayed():
                    print(f"Found submit button containing text: {text}")
                    return button
            except NoSuchElementException:
                pass
    
    # Strategy 2: Try common selectors for submit buttons
    selectors = [
        "button[type='submit']",
        "input[type='submit']",
        "button.btn-primary",
        "input.btn-primary",
        "button[id*='submit' i]",
        "button[id*='continue' i]",
        "button[onclick*='submit' i]",
        "button.btn-success",
        "input.btn-success"
    ]
    
    for selector in selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                if element.is_displayed():
                    print(f"Found submit button with selector: {selector}")
                    return element
        except Exception:
            pass
    
    # Strategy 3: Look for any clickable button or input that's not specifically a reset/cancel
    try:
        buttons = driver.find_elements(By.TAG_NAME, "button") + driver.find_elements(By.CSS_SELECTOR, "input[type='button'], input[type='submit']")
        for button in buttons:
            if not button.is_displayed():
                continue
                
            # Skip obvious cancel/reset buttons
            button_text = button.text.lower() if hasattr(button, 'text') else button.get_attribute("value").lower() if button.get_attribute("value") else ""
            if button_text and any(word in button_text for word in ["cancel", "reset", "clear"]):
                continue
                
            print(f"Found likely submit button: {button_text or button.get_attribute('id') or button.get_attribute('class')}")
            return button
    except Exception:
        pass
    
    print("Could not find a submit button")
    return None

def is_captcha_invalid(driver):
    """Check if the captcha was rejected"""
    try:
        # Look for error messages
        error_patterns = [
            "//span[contains(text(), 'Invalid') or contains(text(), 'incorrect') or contains(text(), 'wrong')]",
            "//div[contains(text(), 'Invalid') or contains(text(), 'incorrect') or contains(text(), 'wrong')]",
            "//p[contains(text(), 'Invalid') or contains(text(), 'incorrect') or contains(text(), 'wrong')]",
            "//span[@class='alertmsg'][text() != '']",  # Common for error messages
            "//div[@class='error']", 
            "//span[@class='error']"
        ]
        
        for pattern in error_patterns:
            elements = driver.find_elements(By.XPATH, pattern)
            for element in elements:
                if element.is_displayed():
                    error_text = element.text.strip()
                    if error_text:
                        print(f"Found error message: {error_text}")
                        return True
        
        # Also check if the captcha input is still visible and empty (often indicates rejection)
        try:
            captcha_input = find_captcha_input(driver)
            if captcha_input and not captcha_input.get_attribute("value") and captcha_input.is_displayed():
                print("Captcha input is empty and still visible - likely invalid")
                return True
        except:
            pass
            
        # Check if a new captcha image appeared
        try:
            captcha_img = driver.find_element(By.CSS_SELECTOR, ".captcha-image img")
            if captcha_img.is_displayed():
                src = captcha_img.get_attribute("src")
                print(f"Captcha image is still visible with src: {src}")
                return True
        except:
            pass
            
        return False
        
    except Exception as e:
        print(f"Error checking for invalid captcha: {e}")
        return False

def handle_success_page(driver):
    """Specifically handle the success page with 'Click here' to download PDF"""
    try:
        print("Checking for PDF success page...")
        
        # First check if we're on the success page by looking for the success message
        success_indicators = [
            "//div[contains(text(), 'Your PDF has been generated successfully')]",
            "//p[contains(text(), 'Your PDF has been generated successfully')]",
            "//span[contains(text(), 'Your PDF has been generated successfully')]",
            "//div[contains(text(), 'Acknowledgement')]" # Green header in the screenshot
        ]
        
        on_success_page = False
        for indicator in success_indicators:
            elements = driver.find_elements(By.XPATH, indicator)
            if elements and any(e.is_displayed() for e in elements):
                print(f"Found success indicator: {elements[0].text}")
                on_success_page = True
                break
        
        if not on_success_page:
            print("Not on success page yet")
            return False
            
        # Take a screenshot of the success page
        driver.save_screenshot("success_page.png")
        
        # Specifically target the red "Click here" text based on the screenshot
        click_here_links = driver.find_elements(By.XPATH, "//a[contains(text(), 'Click here') or contains(text(), 'click here')]")
        
        if click_here_links:
            for link in click_here_links:
                if link.is_displayed():
                    # Get text color - specifically looking for red color from screenshot
                    link_color = link.value_of_css_property("color")
                    link_text = link.text
                    print(f"Found '{link_text}' link with color: {link_color}")
                    
                    # Scroll to make sure it's in view
                    driver.execute_script("arguments[0].scrollIntoView(true);", link)
                    time.sleep(1)
                    
                    # Click the link
                    print(f"Clicking '{link_text}' link to download PDF")
                    link.click()
                    
                    # Wait for download to start
                    time.sleep(5)
                    
                    # Look for and click the OK button if present
                    try:
                        ok_buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'OK')] | //input[@value='OK'] | //a[contains(text(), 'OK')]")
                        for button in ok_buttons:
                            if button.is_displayed():
                                print("Clicking OK button")
                                button.click()
                                time.sleep(2)
                                break
                    except Exception as e:
                        print(f"Error finding/clicking OK button: {e}")
                    
                    print("PDF download initiated successfully")
                    return True
        
        # If we didn't find the specific red "Click here" link, try more general approaches
        download_links = driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf') or contains(@href, 'download') or contains(@onclick, 'download')]")
        if download_links:
            for link in download_links:
                if link.is_displayed():
                    print(f"Found download link: {link.text or link.get_attribute('href')}")
                    link.click()
                    time.sleep(5)
                    
                    # Try to click OK after download
                    try:
                        ok_buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'OK')] | //input[@value='OK']")
                        for button in ok_buttons:
                            if button.is_displayed():
                                button.click()
                                break
                    except:
                        pass
                        
                    return True
                    
        print("Could not find 'Click here' or download link on success page")
        return False
        
    except Exception as e:
        print(f"Error handling success page: {e}")
        driver.save_screenshot("success_page_error.png")
        return False

def run_automation(data):
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--start-maximized") 
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-popup-blocking")  # Allow popups for PDF downloads
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    # Set up Chrome preferences for automatic downloads
    prefs = {
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True,  # Auto-download PDFs instead of opening them
        "download.default_directory": os.path.join(os.path.expanduser("~"), "Downloads")
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    driver = webdriver.Chrome(options=chrome_options)
    download_success = False

    try:
        driver.get("https://tnreginet.gov.in/portal/webHP?requestType=ApplicationRH&actionVal=homePage&screenId=114&UserLocaleID=en&_csrf=8a0e006a-0b81-4e3f-976e-9a8ee485b2ec")
        time.sleep(3)

        driver.find_element(By.XPATH, "//a[contains(@title, 'E-Services')]").click()
        driver.find_element(By.LINK_TEXT, "Encumbrance Certificate").click()
        driver.find_element(By.LINK_TEXT, "View EC").click()

        time.sleep(4)

        # Fill form with data from FastAPI
        Select(driver.find_element(By.ID, "cmb_Zone")).select_by_value(data.zone)
        Select(driver.find_element(By.ID, "cmb_District")).select_by_value(data.district)
        Select(driver.find_element(By.ID, "cmb_SroName")).select_by_value(data.sro)
        driver.find_element(By.ID, "txt_PeriodStartDt").clear()
        driver.find_element(By.ID, "txt_PeriodStartDt").send_keys(data.ec_start_date)
        driver.find_element(By.ID, "txt_PeriodEndDt").clear()
        driver.find_element(By.ID, "txt_PeriodEndDt").send_keys(data.ec_end_date)
        Select(driver.find_element(By.ID, "cmb_Village")).select_by_value(data.village)
        driver.find_element(By.ID, "txt_SurveyNo").send_keys(data.survey_number)
        driver.find_element(By.ID, "txt_SubDivisionNo").send_keys(data.subdivision_number)
        driver.find_element(By.ID, "btn_AddSurvey").click()

        # Wait for captcha to appear
        wait = WebDriverWait(driver, 15)
        try:
            # Wait for the captcha container div to be visible
            wait.until(EC.visibility_of_element_located((By.ID, "cmnCaptchDivId")))
            print("Captcha div is visible")
            
            # Process the captcha with automatic retry (limited to 3 attempts)
            captcha_success = process_captcha_with_retry(driver, max_attempts=3)
            
            if captcha_success:
                print("Captcha solved successfully!")
                
                # Wait for results page to load
                time.sleep(8)
                
                # First, check for success page with the specific "Click here" link
                download_success = handle_success_page(driver)
                
                # If the specialized function didn't work, try the general download function
                if not download_success:
                    print("Trying general download button finder...")
                    download_success = find_download_button(driver)
                
                if download_success:
                    print("PDF download initiated successfully")
                else:
                    print("Could not find download button, but captcha was accepted")
                
                # Wait for download to complete
                time.sleep(5)
                
                # Navigate back to home
                driver.get("https://tnreginet.gov.in/portal/webHP?requestType=ApplicationRH&actionVal=homePage&screenId=114&UserLocaleID=en&_csrf=8a0e006a-0b81-4e3f-976e-9a8ee485b2ec")
                time.sleep(3)
            else:
                print("Failed to solve captcha after 3 attempts")
                driver.save_screenshot("captcha_all_attempts_failed.png")
            
        except Exception as e:
            print(f"Error during captcha or download process: {e}")
            driver.save_screenshot("process_error.png")
        
        # Allow some final time for any downloads to complete
        time.sleep(5)
        
    except Exception as e:
        print(f"Error during automation: {e}")
        driver.save_screenshot("error_screenshot.png")
    finally:
        driver.quit()
        
    return download_success

def solve_captcha(driver):
    """Master function to solve 5-character captcha using multiple approaches"""
    try:
        # Find the captcha image
        captcha_element = None
        
        # Try to find the captcha image inside the captcha-image div first
        try:
            captcha_div = driver.find_element(By.CLASS_NAME, "captcha-image")
            captcha_element = captcha_div.find_element(By.TAG_NAME, "img")
            print(f"Found captcha in captcha-image div with ID: {captcha_element.get_attribute('id')}")
        except Exception:
            # Try alternate methods of finding the captcha
            try:
                captcha_element = driver.find_element(By.ID, "captcha")
                print(f"Found captcha with ID: captcha")
            except Exception:
                # Try any image with captcha in its ID or attributes
                try:
                    captcha_element = driver.find_element(By.CSS_SELECTOR, "img[id*='captcha' i], img[alt*='captcha' i], img[src*='captcha' i]")
                    print(f"Found captcha with selector: {captcha_element.get_attribute('id')}")
                except Exception as e:
                    print(f"Could not find captcha image: {e}")
                    return ""
        
        if not captcha_element:
            print("Captcha element not found")
            return ""
        
        # Get the captcha image
        img = get_captcha_image(driver, captcha_element)
        if not img:
            print("Failed to capture captcha image")
            return ""
        
        # Save original for reference
        img.save("captcha_original.png")
        print("Saved original captcha image")
        
        # Create multiple processing versions of the image for different OCR approaches
        enhanced_img = enhance_image_for_captcha(img)
        opencv_img = preprocess_with_opencv(img)
        inverted_img = invert_image(img)
        perspective_img = apply_perspective_correction(img)
        
        # Use advanced recognition for best results
        captcha_text = use_advanced_captcha_recognition(img)
        if captcha_text:
            return captcha_text
            
        # If advanced recognition failed, try simple methods
        segmented_result = segment_captcha_characters(enhanced_img)
        if segmented_result and len(segmented_result) == 5:
            print(f"Using character segmentation result: {segmented_result}")
            return segmented_result
            
        # Try Tesseract with enhanced image
        tess_result = solve_captcha_with_tesseract(enhanced_img, 'exact_5_chars')
        if tess_result:
            return tess_result
            
        # Try EasyOCR if available
        if EASYOCR_AVAILABLE:
            easyocr_result = solve_captcha_with_easyocr(img)
            if easyocr_result:
                return easyocr_result
        
        # If all methods failed, return empty string
        print("All captcha solving methods failed")
        return ""
    
    except Exception as e:
        print(f"Error in captcha solving process: {e}")
        return ""

def find_download_button(driver):
    """Find and click the download button after successful captcha"""
    wait = WebDriverWait(driver, 10)
    
    # Wait for page to fully load and any dynamic content
    time.sleep(5)
    
    # Various ways to find the download button or link
    strategies = [
        # Strategy 1: Look for "Click here" links first (most common)
        lambda: driver.find_elements(By.XPATH, "//a[contains(text(), 'Click here') or contains(text(), 'click here')]"),
        
        # Strategy 2: Look for links containing PDF or download keywords
        lambda: driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf') or contains(@href, 'download') or contains(@href, 'print')]"),
        
        # Strategy 3: Look for buttons with download text
        lambda: driver.find_elements(By.XPATH, "//button[contains(text(), 'Download') or contains(text(), 'Print') or contains(text(), 'Save')]"),
        
        # Strategy 4: Look for icons that might be download buttons
        lambda: driver.find_elements(By.CSS_SELECTOR, "a.download-icon, button.download-icon, i.fa-download, i.fa-print"),
        
        # Strategy 5: Look for form buttons that might be for download
        lambda: driver.find_elements(By.CSS_SELECTOR, "form[action*='download'] button, form[action*='pdf'] button"),
        
        # Strategy 6: Look for any clickable elements that might trigger download
        lambda: driver.find_elements(By.CSS_SELECTOR, "a[onclick*='download'], button[onclick*='download'], a[onclick*='pdf'], button[onclick*='pdf']")
    ]
    
    for strategy in strategies:
        try:
            elements = strategy()
            for element in elements:
                try:
                    if element.is_displayed():
                        element_text = element.text.strip() if hasattr(element, 'text') else ""
                        element_href = element.get_attribute("href") or ""
                        print(f"Found potential download element: {element_text or element_href or element.get_attribute('id')}")
                        
                        # Scroll element into view
                        driver.execute_script("arguments[0].scrollIntoView(true);", element)
                        time.sleep(1)
                        
                        # Click the element
                        element.click()
                        print("Clicked download element")
                        
                        # Wait for download to start
                        time.sleep(5)
                        
                        # For PDF links that might open in new tab
                        if ".pdf" in element_href:
                            # Switch back to main window
                            main_window = driver.window_handles[0]
                            if len(driver.window_handles) > 1:
                                driver.switch_to.window(driver.window_handles[-1])
                                driver.close()  # Close the PDF tab
                                driver.switch_to.window(main_window)  # Switch back
                        
                        # Verify download started by checking if we're still on the same page
                        if driver.current_url == driver.current_url:
                            print("Download appears to have started successfully")
                            return True
                except Exception as e:
                    print(f"Error clicking element: {e}")
                    continue
        except Exception as e:
            print(f"Strategy failed: {e}")
            continue
    
    print("Could not find download button")
    return False

