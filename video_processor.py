import cv2
import torch
import numpy as np
from PIL import Image
from threading import Thread, Lock
from queue import Queue
import base64
import time

class SRCNN(torch.nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = torch.nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class UNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

def center_crop(tensor, target_size):
    _, _, h, w = tensor.size()
    target_h, target_w = target_size
    delta_h = h - target_h
    delta_w = w - target_w
    top = delta_h // 2
    left = delta_w // 2
    return tensor[:, :, top:top+target_h, left:left+target_w]

class UNet(torch.nn.Module):
    def __init__(self, input_channels=6, output_channels=3):
        super(UNet, self).__init__()
        self.enc1 = UNetBlock(input_channels, 64)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.enc2 = UNetBlock(64, 128)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(128, 256)
        self.up1 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(256, 128)
        self.up2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(128, 64)
        self.final_conv = torch.nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # Add padding to ensure output size matches input size
        pad = 4  # Adjust padding size based on your network architecture
        x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='reflect')
        
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        up1 = self.up1(bottleneck)
        enc2_cropped = center_crop(enc2, up1.shape[2:])
        dec1 = self.dec1(torch.cat([up1, enc2_cropped], dim=1))
        up2 = self.up2(dec1)
        enc1_cropped = center_crop(enc1, up2.shape[2:])
        dec2 = self.dec2(torch.cat([up2, enc1_cropped], dim=1))
        output = self.final_conv(dec2)
        
        # Crop the output to match the input size
        output = center_crop(output, (x.size(2) - 2*pad, x.size(3) - 2*pad))
        return output

def sharpen_image(image_np):
    # Stronger sharpening kernel
    kernel = np.array([[-1, -1, -1],
                      [-1, 9, -1],
                      [-1, -1, -1]])
    sharpened = cv2.filter2D(image_np, -1, kernel)
    return sharpened

def detect_edges(image_np):
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    # Convert back to RGB for visualization
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def gamma_correction(image_np, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image_np, table)

def enhance_saturation(image_np, scale=1.2):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * scale, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def apply_bilateral_filter(image_np, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(image_np, d, sigmaColor, sigmaSpace)

def post_process_sequence2(image):
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Apply comprehensive post-processing pipeline
    img_array = sharpen_image(img_array)
    img_array = gamma_correction(img_array, gamma=1.1)
    img_array = enhance_saturation(img_array)
    img_array = apply_bilateral_filter(img_array)
    
    # Ensure brightness preservation
    mean_brightness_orig = np.mean(img_array)
    corrected = img_array.astype(np.float32)
    mean_brightness_corr = np.mean(corrected)
    brightness_factor = mean_brightness_orig / mean_brightness_corr if mean_brightness_corr > 0 else 1
    corrected = np.clip(corrected * brightness_factor, 0, 255)
    
    return Image.fromarray(np.uint8(corrected))

class VideoProcessor:
    def __init__(self, srcnn_model, unet_model):
        self.srcnn_model = srcnn_model
        self.unet_model = unet_model
        self.frame_queue = Queue(maxsize=3)
        self.result_queue = Queue(maxsize=3)
        self.lock = Lock()
        self.is_running = False
        self.processing_thread = None

    def start_processing(self):
        self.is_running = True
        self.processing_thread = Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def stop_processing(self):
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()

    def _process_frames(self):
        while self.is_running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                print("Processing frame with shape:", frame.shape if frame is not None else None)
                enhanced_frame = self.enhance_frame(frame)
                if enhanced_frame is not None:
                    print("Enhanced frame generated with shape:", enhanced_frame.shape)
                    if not self.result_queue.full():
                        self.result_queue.put(enhanced_frame)
                    else:
                        print("Result queue is full, skipping frame")
                else:
                    print("Frame enhancement failed")

    def add_frame(self, frame):
        if not self.frame_queue.full():
            self.frame_queue.put(frame)

    def get_enhanced_frame(self):
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None

    def enhance_frame(self, frame):
        try:
            # Start timing the process
            start_time = time.time()
            # Handle base64 encoded image data
            if isinstance(frame, str) and frame.startswith('data:image'):
                try:
                    # Extract the base64 data
                    _, encoded = frame.split(',', 1)
                    # Convert base64 to bytes
                    img_bytes = base64.b64decode(encoded)
                    # Convert to numpy array
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    # Decode image
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                except Exception as e:
                    print(f"Error decoding base64 image: {str(e)}")
                    return None

            # Validate input frame
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                print("Invalid frame input: frame is None or invalid type")
                return None

            # Validate frame dimensions and type
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                print(f"Invalid frame format: expected 3-channel color image, got shape {frame.shape}")
                return None

            print(f"Starting enhancement of frame with shape {frame.shape}")

            # Make a copy of the frame to avoid modifying the original
            frame = frame.copy()
            
            # Resize frame to reduce processing time and memory usage
            height, width = frame.shape[:2]
            scale_factor = min(1.0, 512 / max(height, width))
            if scale_factor < 1.0:
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Convert frame to RGB
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                print("Successfully converted frame to RGB")
            except Exception as e:
                print(f"Error converting frame to RGB: {str(e)}")
                return None

            # Convert to tensor with optimized memory usage
            try:
                # Convert to float32 and normalize before creating tensor
                frame_rgb = frame_rgb.astype(np.float32) / 255.0
                frame_rgb = np.ascontiguousarray(frame_rgb.transpose(2, 0, 1))
                img_tensor = torch.from_numpy(frame_rgb).unsqueeze(0).float()
                img_tensor = img_tensor.contiguous()
                print("Successfully converted frame to tensor with shape:", img_tensor.shape)

                # Process with models
                with torch.no_grad():
                    try:
                        # SRCNN enhancement
                        srcnn_output = self.srcnn_model(img_tensor)
                        if srcnn_output is None:
                            raise RuntimeError("SRCNN model failed to process frame")
                        print("SRCNN enhancement completed successfully")

                        # Ensure tensor dimensions are valid
                        if srcnn_output.shape != img_tensor.shape:
                            srcnn_output = torch.nn.functional.interpolate(srcnn_output, size=img_tensor.shape[2:], mode='bilinear', align_corners=False)

                        # UNet enhancement
                        unet_input = torch.cat([img_tensor, srcnn_output], dim=1)
                        enhanced_output = self.unet_model(unet_input)
                        print("UNet enhancement completed successfully")

                        # Convert back to numpy array
                        enhanced_frame = enhanced_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        enhanced_frame = np.clip(enhanced_frame * 255, 0, 255).astype(np.uint8)

                        # Apply post-processing pipeline with edge detection
                        try:
                            # Store original frame and calculate its edges
                            original_frame = frame.copy()
                            original_edges = detect_edges(original_frame)
                            original_edges_base64 = base64.b64encode(cv2.imencode('.jpg', original_edges)[1]).decode('utf-8')
                            
                            # Apply enhanced sharpening and other effects
                            enhanced_frame = sharpen_image(enhanced_frame)
                            enhanced_frame = gamma_correction(enhanced_frame)
                            enhanced_frame = enhance_saturation(enhanced_frame)
                            enhanced_frame = apply_bilateral_filter(enhanced_frame)
                            
                            # Calculate edge detection for enhanced frame
                            enhanced_edges = detect_edges(enhanced_frame)
                            enhanced_edges_base64 = base64.b64encode(cv2.imencode('.jpg', enhanced_edges)[1]).decode('utf-8')
                            
                            # Store metrics and edge detection results
                            metrics = {
                                'original_edges': original_edges_base64,
                                'enhanced_edges': enhanced_edges_base64,
                                'process_time': time.time() - start_time
                            }
                            
                            # Create response data structure
                            response_data = {
                                'frame': enhanced_frame,
                                'metrics': metrics
                            }
                            enhanced_frame = response_data
                            
                        except Exception as post_err:
                            print(f"Post-processing error: {str(post_err)}")
                            # Continue with original enhanced frame if post-processing fails
                            enhanced_frame = {'frame': enhanced_frame, 'metrics': {}}
                            pass

                        # Resize back to original dimensions if needed
                        if scale_factor < 1.0 and isinstance(enhanced_frame, dict) and 'frame' in enhanced_frame:
                            enhanced_frame['frame'] = cv2.resize(enhanced_frame['frame'], (width, height), interpolation=cv2.INTER_LANCZOS4)

                        print(f"Successfully completed frame enhancement, output shape: {enhanced_frame['frame'].shape if isinstance(enhanced_frame, dict) and 'frame' in enhanced_frame else enhanced_frame.shape}")

                        # Return the enhanced frame with metrics
                        return response_data
                    except Exception as model_error:
                        print(f"Model processing error: {str(model_error)}")
                        # Clean up any tensors that might exist
                        if 'img_tensor' in locals(): del img_tensor
                        if 'srcnn_output' in locals(): del srcnn_output
                        if 'unet_input' in locals(): del unet_input
                        if 'enhanced_output' in locals(): del enhanced_output
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        return None

                    try:
                        # Apply post-processing with proper color space handling
                        if isinstance(enhanced_output, dict):
                            if 'frame' in enhanced_output:
                                enhanced_output = enhanced_output['frame']
                            else:
                                print("No frame data found in enhanced output")
                                return None
                        
                        if not isinstance(enhanced_output, np.ndarray):
                            print(f"Invalid enhanced output type: {type(enhanced_output)}")
                            return None
                            
                        enhanced_image = Image.fromarray(enhanced_output, mode='RGB')
                        enhanced_image = post_process_sequence2(enhanced_image)
                        del enhanced_output

                        # Convert back to BGR with proper color space handling
                        enhanced_array = np.array(enhanced_image)
                        if enhanced_array.shape[-1] != 3:
                            print(f"Invalid enhanced image format: {enhanced_array.shape}")
                            return None
                        enhanced_frame = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2BGR)
                        if scale_factor < 1.0:
                            enhanced_frame = cv2.resize(enhanced_frame, (width, height), interpolation=cv2.INTER_LINEAR)
                        # Ensure valid frame format
                        if not (enhanced_frame.shape[-1] == 3 and enhanced_frame.dtype == np.uint8):
                            print(f"Invalid frame format after conversion: shape={enhanced_frame.shape}, dtype={enhanced_frame.dtype}")
                            return None
                        
                        print(f"Successfully completed frame enhancement, output shape: {enhanced_frame.shape}")
                        # Convert the enhanced frame to base64
                        try:
                            # Ensure the frame is in the correct format for encoding
                            if not isinstance(enhanced_frame, np.ndarray):
                                print(f"Invalid frame type for encoding: {type(enhanced_frame)}")
                                return None
                                
                            # Verify frame dimensions and type before encoding
                            if len(enhanced_frame.shape) != 3 or enhanced_frame.shape[2] != 3:
                                print(f"Invalid frame format for encoding: {enhanced_frame.shape}")
                                return None
                                
                            # Ensure frame is uint8 type
                            if enhanced_frame.dtype != np.uint8:
                                print(f"Converting frame from {enhanced_frame.dtype} to uint8")
                                enhanced_frame = np.clip(enhanced_frame, 0, 255).astype(np.uint8)
                                
                            # Encode the frame as JPEG with quality parameter (try multiple quality levels if needed)
                            for quality in [95, 90, 85, 80]:
                                try:
                                    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                                    success, buffer = cv2.imencode('.jpg', enhanced_frame, encode_params)
                                    if success and buffer is not None and buffer.size > 0:
                                        break
                                    print(f"Encoding at quality {quality} failed, trying lower quality")
                                except Exception as encode_err:
                                    print(f"Error during encoding at quality {quality}: {str(encode_err)}")
                                    continue
                            
                            if not success or buffer is None or buffer.size == 0:
                                print("Failed to encode frame as JPEG at any quality level")
                                return None
                                
                            # Convert to base64 and include edge detection results
                            try:
                                img_base64 = base64.b64encode(buffer).decode('utf-8')
                                # Create data URL with metrics
                                response_data = {
                                    'frame': f'data:image/jpeg;base64,{img_base64}',
                                    'metrics': metrics if isinstance(metrics, dict) else {}
                                }
                                print(f"Successfully encoded frame and edges to base64")
                                return response_data
                            except Exception as b64_err:
                                print(f"Error in base64 encoding: {str(b64_err)}")
                                # If base64 encoding fails, return None to indicate failure
                                print("Base64 encoding failed")
                                return None
                        except Exception as e:
                            print(f"Error encoding output frame: {str(e)}")
                            return None
                    except Exception as post_error:
                        print(f"Post-processing error: {str(post_error)}")
                        if 'enhanced_output' in locals(): del enhanced_output
                        return None

            except torch.cuda.OutOfMemoryError:
                print("CUDA out of memory error - clearing cache")
                torch.cuda.empty_cache()
                return None
            except Exception as e:
                print(f"Error in model inference: {str(e)}")
                return None
            finally:
                # Ensure memory is cleaned up
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"Error in enhance_frame: {str(e)}")
            return None