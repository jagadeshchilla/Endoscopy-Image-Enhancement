# Endoscopy Image Enhancement 🔬

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Flask](https://img.shields.io/badge/Framework-Flask-000000.svg?logo=flask)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-%23white.svg?logo=opencv&logoColor=white)](https://opencv.org/)

> An advanced medical image enhancement web application leveraging deep learning models for improved diagnostic visualization.

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge&logo=git"/>
</p>

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Technical Architecture](#technical-architecture)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Models](#models)
- [Performance Metrics](#performance-metrics)
- [Tech Stack](#tech-stack)
- [Contributors](#contributors)
- [License](#license)

## 🔭 Overview

This web application provides a state-of-the-art interface for enhancing medical endoscopy images using advanced AI models. It supports both image and video enhancement with real-time processing capabilities, making it valuable for medical professionals in diagnostic procedures.

## ⭐ Key Features

| Feature | Description |
|---------|------------|
| 🖼️ Image Enhancement | Real-time enhancement using SRCNN and UNet models |
| 🎯 Edge Detection | Advanced visualization of edges in original and enhanced images |
| 🎥 Video Processing | Support for video file processing and webcam integration |
| 📸 Frame Capture | Intelligent frame capture and enhancement from video streams |
| 📊 Analytics | Real-time PSNR and SSIM metrics visualization |

## 🏗️ Technical Architecture

```mermaid
graph TD
    A[User Input] --> B[Web Interface]
    B --> C{Processing Type}
    C -->|Image| D[SRCNN Model]
    C -->|Video| E[UNet Model]
    D --> F[Enhancement Pipeline]
    E --> F
    F --> G[Edge Detection]
    F --> H[Metrics Calculation]
    G --> I[Results Display]
    H --> I
```

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/Kedhareswer/Endoscopy-Image-Enhancement.git
cd Endoscopy-Image-Enhancement

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 📖 Usage Guide

### Image Enhancement
1. Navigate to the "Image Enhancement" tab
2. Upload an image (drag & drop or browse)
3. View enhanced results and metrics

### Video Enhancement
1. Access the "Video Enhancement" section
2. Choose video input source
3. Use controls for frame capture
4. Review enhanced frames

## 🧠 Models

### SRCNN (Super-Resolution CNN)
- Architecture: 3-layer CNN
- Activation: ReLU
- Purpose: Image super-resolution
- Input: Low-resolution medical images
- Output: Enhanced high-resolution images

### UNet
- Architecture: Encoder-decoder with skip connections
- Purpose: Biomedical image segmentation
- Speciality: Spatial information preservation
- Application: Real-time video enhancement

## 📊 Performance Metrics

| Metric | Description | Implementation |
|--------|-------------|----------------|
| PSNR | Peak Signal-to-Noise Ratio | Image quality assessment |
| SSIM | Structural Similarity Index | Structure preservation measurement |

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| Backend | ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white) |
| Deep Learning | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) |
| Image Processing | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) |
| Frontend | ![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black) |

## 👥 Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Kedhareswer">
        <img src="https://avatars.githubusercontent.com/Kedhareswer" width="100px;" alt="Kedhareswer"/>
        <br />
        <sub><b>Kedhareswer</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/jagadeshchilla">
        <img src="https://avatars.githubusercontent.com/jagadeshchilla" width="100px;" alt="Jagadesh Chilla"/>
        <br />
        <sub><b>Jagadesh Chilla</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://linkedin.com/in/manish-chetla">
        <img src="https://avatars.githubusercontent.com/u/default" width="100px;" alt="Manish Chetla"/>
        <br />
        <sub><b>Manish Chetla</b></sub>
      </a>
    </td>
  </tr>
</table>

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SRCNN architecture: Chao Dong et al.
- UNet architecture: Olaf Ronneberger et al.
- Medical imaging community for valuable feedback

---
<p align="center">Made with ❤️ for better medical imaging</p>
