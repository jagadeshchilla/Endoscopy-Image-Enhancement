import os
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify, Response
from werkzeug.utils import secure_filename
import cv2
import base64
import uuid
import os
from skimage import exposure
from video_processor import VideoProcessor, SRCNN, UNet

app = Flask(__name__)

@app.route('/model_info')
def model_info():
    return render_template('model_info.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize video processor and capture variables
video_processor = None
cap = None

# Initialize models
srcnn_model = SRCNN()
unet_model = UNet()

# Load pre-trained weights
srcnn_model.load_state_dict(torch.load('srcnn_full_image.pth', map_location=torch.device('cpu')))
unet_model.load_state_dict(torch.load('unet_full_image.pth', map_location=torch.device('cpu')))

# Set models to evaluation mode
srcnn_model.eval()
unet_model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def calculate_metrics(original, enhanced):
    # Convert images to numpy arrays
    original_np = np.array(original)
    enhanced_np = np.array(enhanced)
    
    # Calculate PSNR
    mse = np.mean((original_np - enhanced_np) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Calculate SSIM
    def ssim(img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = np.ones((11,11))/121
        mu1 = np.apply_over_axes(np.mean, img1, (-3,-2))
        mu2 = np.apply_over_axes(np.mean, img2, (-3,-2))
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1*mu2
        sigma1_sq = np.apply_over_axes(np.mean, img1**2, (-3,-2)) - mu1_sq
        sigma2_sq = np.apply_over_axes(np.mean, img2**2, (-3,-2)) - mu2_sq
        sigma12 = np.apply_over_axes(np.mean, img1*img2, (-3,-2)) - mu1_mu2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return np.mean(ssim_map)
    
    ssim_value = ssim(original_np, enhanced_np)
    return psnr, ssim_value

def apply_gamma_correction(image, gamma=1.1):
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    # Apply gamma correction with brightness preservation
    corrected = exposure.adjust_gamma(img_array, gamma)
    # Ensure brightness preservation
    mean_brightness_orig = np.mean(img_array)
    mean_brightness_corr = np.mean(corrected)
    brightness_factor = mean_brightness_orig / mean_brightness_corr if mean_brightness_corr > 0 else 1
    corrected = np.clip(corrected * brightness_factor, 0, 255)
    return Image.fromarray(np.uint8(corrected))

def detect_edges(image):
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Multi-scale bilateral filtering for enhanced noise reduction
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    denoised = cv2.bilateralFilter(denoised, d=7, sigmaColor=50, sigmaSpace=50)
    denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
    
    # Multi-scale edge detection
    edges_list = []
    scales = [0.5, 1.0, 2.0]
    
    for scale in scales:
        # Resize image according to scale
        if scale != 1.0:
            scaled = cv2.resize(denoised, None, fx=scale, fy=scale)
        else:
            scaled = denoised
            
        # Canny edge detection with automatic thresholding
        median = np.median(scaled)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        canny = cv2.Canny(scaled, lower, upper)
        
        # Resize back to original size
        if scale != 1.0:
            canny = cv2.resize(canny, (denoised.shape[1], denoised.shape[0]))
        
        edges_list.append(canny)
    
    # Combine multi-scale edges
    multi_scale_edges = np.maximum.reduce(edges_list)
    
    # Enhanced Sobel edge detection
    sobelx = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Non-maximum suppression
    direction = np.arctan2(sobely, sobelx) * 180 / np.pi
    direction = np.abs(direction)
    
    nms = np.zeros_like(magnitude)
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            if (0 <= direction[i,j] < 22.5) or (157.5 <= direction[i,j] <= 180):
                if magnitude[i,j] >= magnitude[i,j-1] and magnitude[i,j] >= magnitude[i,j+1]:
                    nms[i,j] = magnitude[i,j]
            elif (22.5 <= direction[i,j] < 67.5):
                if magnitude[i,j] >= magnitude[i-1,j+1] and magnitude[i,j] >= magnitude[i+1,j-1]:
                    nms[i,j] = magnitude[i,j]
            elif (67.5 <= direction[i,j] < 112.5):
                if magnitude[i,j] >= magnitude[i-1,j] and magnitude[i,j] >= magnitude[i+1,j]:
                    nms[i,j] = magnitude[i,j]
            else:
                if magnitude[i,j] >= magnitude[i-1,j-1] and magnitude[i,j] >= magnitude[i+1,j+1]:
                    nms[i,j] = magnitude[i,j]
    
    # Normalize and enhance Sobel edges
    sobel_edges = np.uint8(255 * nms / np.max(nms))
    
    # Combine edge detection results
    edges = cv2.addWeighted(multi_scale_edges, 0.6, sobel_edges, 0.4, 0)
    
    # Apply morphological operations to clean up edges
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Enhance contrast of edges
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
    _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    
    # Convert back to RGB for consistent display
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

def apply_sharpening(image, amount=1.2):
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(img_array, 5, 50, 50)
    # Create blurred version using Gaussian blur with smaller sigma
    blurred = cv2.GaussianBlur(denoised, (0, 0), 2)
    # Apply unsharp mask with moderate amount
    sharpened = cv2.addWeighted(denoised, 1.0 + amount, blurred, -amount, 0)
    # Use gentler high-pass filter for fine details
    kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]]) / 1.0
    sharpened = cv2.filter2D(sharpened, -1, kernel)
    # Ensure output stays within valid range
    sharpened = np.clip(sharpened, 0, 255)
    return Image.fromarray(np.uint8(sharpened))

def enhance_saturation(image, factor=1.3):
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    # Convert to HSV
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
    # Enhance saturation
    hsv[:, :, 1] = hsv[:, :, 1] * factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    # Convert back to RGB
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return Image.fromarray(enhanced)

def post_process_sequence1(image):
    # Gamma correction -> Sharpening -> Saturation
    gamma_corrected = apply_gamma_correction(image)
    sharpened = apply_sharpening(gamma_corrected)
    saturated = enhance_saturation(sharpened)
    return saturated

def post_process_sequence2(image):
    # Saturation -> Gamma correction -> Sharpening
    saturated = enhance_saturation(image)
    gamma_corrected = apply_gamma_correction(saturated)
    sharpened = apply_sharpening(gamma_corrected)
    return sharpened

def process_image(image_path):
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        
        # Resize image to 299x299 to match model input size
        img_resized = img.resize((299, 299), Image.Resampling.LANCZOS)
        
        # Convert image to tensor and normalize
        img_tensor = torch.from_numpy(np.array(img_resized)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Process with both models
        with torch.no_grad():
            try:
                # SRCNN enhancement
                srcnn_output = srcnn_model(img_tensor)
                if srcnn_output is None:
                    raise RuntimeError("SRCNN model failed to process image")
                
                # Ensure tensor dimensions are valid
                if srcnn_output.shape != img_tensor.shape:
                    srcnn_output = torch.nn.functional.interpolate(srcnn_output, size=img_tensor.shape[2:], mode='bilinear', align_corners=False)
                
                # UNet enhancement
                unet_input = torch.cat([img_tensor, srcnn_output], dim=1)
                enhanced_output = unet_model(unet_input)
                
                # Convert enhanced output back to image
                enhanced_output = enhanced_output.squeeze(0).permute(1, 2, 0)
                enhanced_output = np.clip(enhanced_output.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
                enhanced_image = Image.fromarray(enhanced_output)
                
                # Resize back to original dimensions
                enhanced_image = enhanced_image.resize(original_size, Image.Resampling.LANCZOS)
                
                # Apply post-processing
                enhanced_image = apply_sharpening(enhanced_image)
                enhanced_image = apply_gamma_correction(enhanced_image)
                
                # Generate edge detection
                original_edges = detect_edges(img)
                enhanced_edges = detect_edges(enhanced_image)
                
                return enhanced_image, img, original_edges, enhanced_edges
            except Exception as model_error:
                print(f"Model processing error: {str(model_error)}")
                return None, None, None, None
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        return None, None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enhancement')
def enhancement():
    return render_template('enhancement.html')

@app.route('/enhance_frame', methods=['POST'])
def enhance_frame():
    try:
        # Check if request contains file data
        if 'frame' in request.files:
            # Handle FormData with blob
            frame_file = request.files['frame']
            frame_bytes = frame_file.read()
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        elif request.is_json:
            # Handle JSON with base64 data
            frame_data = request.get_json()
            if not frame_data or 'frame' not in frame_data:
                return jsonify({'success': False, 'error': 'No frame data provided'}), 400
            
            try:
                # Convert base64 frame data to numpy array
                frame_str = frame_data['frame'].split(',')[1]
                frame_bytes = np.frombuffer(base64.b64decode(frame_str), np.uint8)
                frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
            except Exception as e:
                return jsonify({'success': False, 'error': f'Frame decoding error: {str(e)}'}), 400
        else:
            return jsonify({'success': False, 'error': 'Unsupported request format'}), 400
            
        if frame is None:
            return jsonify({'success': False, 'error': 'Invalid frame data'}), 400

        # Use global video processor instance
        global video_processor
        if video_processor is None:
            video_processor = VideoProcessor(srcnn_model, unet_model)

        # Process the frame
        result = video_processor.enhance_frame(frame)
        
        if result is None:
            return jsonify({'success': False, 'error': 'Frame enhancement failed'}), 500

        # Extract frame and metrics from result
        if isinstance(result, dict) and 'frame' in result:
            enhanced_frame = result['frame']
            metrics = result.get('metrics', {})
        else:
            enhanced_frame = result
            metrics = {}

        # Calculate PSNR and SSIM if not provided in metrics
        if 'psnr' not in metrics or 'ssim' not in metrics:
            try:
                psnr, ssim = calculate_metrics(Image.fromarray(frame), Image.fromarray(enhanced_frame))
                metrics.update({'psnr': psnr, 'ssim': ssim})
            except Exception as metric_err:
                print(f"Error calculating metrics: {str(metric_err)}")

        # Encode the enhanced frame
        try:
            # Ensure frame is in the correct format
            if not isinstance(enhanced_frame, np.ndarray):
                enhanced_frame = np.array(enhanced_frame)

            # Convert to BGR for OpenCV encoding
            if enhanced_frame.shape[-1] == 3:
                enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)

            # Ensure uint8 type
            if enhanced_frame.dtype != np.uint8:
                enhanced_frame = np.clip(enhanced_frame, 0, 255).astype(np.uint8)

            # Encode frame
            success, buffer = cv2.imencode('.jpg', enhanced_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not success:
                return jsonify({'success': False, 'error': 'Frame encoding failed'}), 500

            # Convert to base64
            enhanced_frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'enhanced_frame': f'data:image/jpeg;base64,{enhanced_frame_b64}',
                'metrics': metrics
            })

        except Exception as e:
            return jsonify({'success': False, 'error': f'Frame encoding error: {str(e)}'}), 500

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/enhance_image', methods=['POST'])
def enhance_image():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = secure_filename(str(uuid.uuid4()) + '_' + file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save uploaded file
            file.save(filepath)
            
            try:
                # Process the image
                enhanced_image, original_image, original_edges, enhanced_edges = process_image(filepath)
                
                if enhanced_image is None:
                    os.remove(filepath)  # Clean up uploaded file
                    return jsonify({'success': False, 'error': 'Image enhancement failed'}), 500
                    
                # Save enhanced version and edge detection results
                enhanced_path = os.path.join(app.config['UPLOAD_FOLDER'], f'enhanced_{filename}')
                original_edges_path = os.path.join(app.config['UPLOAD_FOLDER'], f'edges_original_{filename}')
                enhanced_edges_path = os.path.join(app.config['UPLOAD_FOLDER'], f'edges_enhanced_{filename}')
                
                enhanced_image.save(enhanced_path)
                original_edges.save(original_edges_path)
                enhanced_edges.save(enhanced_edges_path)
                
                # Calculate metrics
                psnr, ssim = calculate_metrics(original_image, enhanced_image)
                
                return jsonify({
                    'success': True,
                    'original': f'/static/uploads/{filename}',
                    'enhanced': f'/static/uploads/enhanced_{filename}',
                    'edges_original': f'/static/uploads/edges_original_{filename}',
                    'edges_enhanced': f'/static/uploads/edges_enhanced_{filename}',
                    'metrics': {
                        'psnr': float(psnr),
                        'ssim': float(ssim)
                    }
                })
            except Exception as process_error:
                # Clean up uploaded file if processing fails
                os.remove(filepath)
                return jsonify({'success': False, 'error': str(process_error)}), 500
        else:
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Initialize video processor
video_processor = VideoProcessor(srcnn_model, unet_model)

@app.route('/process_video_frame', methods=['POST'])
def process_video_frame():
    try:
        # Check if request contains file data
        if 'frame' in request.files:
            # Handle FormData with blob
            frame_file = request.files['frame']
            frame_bytes = frame_file.read()
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        elif request.is_json:
            # Handle JSON with base64 data
            frame_data = request.get_json()
            if not frame_data or 'frame' not in frame_data:
                return jsonify({'success': False, 'error': 'No frame data provided'}), 400
            
            try:
                # Convert base64 frame data to numpy array
                frame_str = frame_data['frame'].split(',')[1]
                frame_bytes = np.frombuffer(base64.b64decode(frame_str), np.uint8)
                frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
            except Exception as e:
                return jsonify({'success': False, 'error': f'Frame decoding error: {str(e)}'}), 400
        else:
            return jsonify({'success': False, 'error': 'Unsupported request format'}), 400
            
        if frame is None:
            return jsonify({'success': False, 'error': 'Invalid frame data'}), 400

        # Process the frame
        enhanced_frame = video_processor.enhance_frame(frame)
        
        if enhanced_frame is None:
            return jsonify({'success': False, 'error': 'Frame enhancement failed'}), 500

        # Calculate metrics if original frame is available
        metrics = {}
        if isinstance(frame, np.ndarray) and isinstance(enhanced_frame, np.ndarray):
            original_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB))
            psnr, ssim = calculate_metrics(original_pil, enhanced_pil)
            metrics = {'psnr': float(psnr), 'ssim': float(ssim)}

        # Convert enhanced frame to base64
        _, buffer = cv2.imencode('.jpg', enhanced_frame)
        enhanced_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'enhanced_frame': f'data:image/jpeg;base64,{enhanced_base64}',
            'metrics': metrics
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global video_processor, cap
        
        if cap is None:
            cap = cv2.VideoCapture(0)  # Open webcam
            video_processor = VideoProcessor(srcnn_model, unet_model)
            video_processor.start_processing()
        
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                
                # Encode original frame for display
                _, orig_buffer = cv2.imencode('.jpg', frame)
                orig_bytes = orig_buffer.tobytes()
                
                # Process frame through video processor
                enhanced_frame = video_processor.enhance_frame(frame)
                
                # Encode enhanced frame if available
                if enhanced_frame is not None:
                    _, enh_buffer = cv2.imencode('.jpg', enhanced_frame)
                    enh_bytes = enh_buffer.tobytes()
                else:
                    enh_bytes = orig_bytes
                
                # Yield both original and enhanced frames
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + orig_bytes + b'\r\n'
                       b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + enh_bytes + b'\r\n')
        except Exception as e:
            print(f"Error in video streaming: {str(e)}")
        finally:
            if video_processor:
                video_processor.stop_processing()
            if cap:
                cap.release()
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video')
def stop_video():
    global video_processor, cap
    if video_processor:
        video_processor.stop_processing()
        video_processor = None
    if cap:
        cap.release()
        cap = None
    return jsonify({'success': True})

# Register static folder for serving uploaded files
app.static_folder = 'static'

if __name__ == '__main__':
    # Use environment variable for port if available (for cloud deployment)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)