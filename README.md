# Fisheye Image Projection Project

A Python computer vision toolkit for processing fisheye camera images with perspective and spherical projection capabilities.

## Overview

This project provides tools for converting fisheye camera images into different projection formats, enabling users to extract specific views or create panoramic images from fisheye input. It follows OpenCV fisheye camera model conventions and provides both educational reference implementations and optimized vectorized algorithms for production use.

## Core Functionalities

### Projection Types
- **Perspective Projection**: Extract rectilinear views from fisheye images (similar to traditional camera perspective)
- **Spherical Projection**: Generate equirectangular panoramic images suitable for VR/360° viewing

### Key Features
- Two fisheye camera image projection classes to generate vitual views
- Caching system for generated projection x, y maps
- Example programs to use the image projection functionalities
- Example interactive GUI programs to show case projection results with virtual camera controls

### GUI Demo

The project includes an interactive GUI application that provides real-time visualization and control over fisheye image projections,
given virtual camera parameter set from the UI.

![GUI Demo Animation](docs/videos/gui_demo.gif)
*Interactive GUI demonstration showing real-time fisheye projection controls*

It supports
* Perspective projection interface with real-time parameter controls
* Spherical projection interface with panoramic view generation

Launch the GUI demo:

```bash
python gui/gui_demo.py
```

## Acknowledgments

We thank the authors of the FIORD dataset for providing the example fisheye camera image used in this project. The fisheye image demonstrates the capabilities of the projection algorithms and serves as a reference for testing and development.

### Citation

If you use the example fisheye image in your research or applications, please cite the following work:

```bibtex
@article{gunes2025fiord,
  author    = {Gunes, Ulas and Turkulainen, Matias and Ren, Xuqian and Solin, Arno and Kannala, Juho and Rahtu, Esa},
  title     = {FIORD: A Fisheye Indoor-Outdoor Dataset with LIDAR Ground Truth for 3D Scene Reconstruction and Benchmarking},
  booktitle = {SCIA},
  year      = {2025},
}
```
