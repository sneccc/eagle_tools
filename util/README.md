# Eagle Tools Utility

This utility package provides a set of tools for processing images and captions from Eagle, a digital asset management software.

## Modules

### caption_utils.py

Utilities for preparing and processing captions and tags.

### image_utils.py

Functions for image processing, including resizing, cropping, and SVG handling.

### io_utils.py

Utility functions for file and directory operations.

### LLM_API.py

An asynchronous processor for generating captions using a Language Model.

### process_image_API.py

Main API for processing images, including resizing, cropping, and caption generation.

## Usage

These utilities are designed to work with the Eagle_extractor.ipynb notebook. They provide the core functionality for extracting and processing images and captions from Eagle packs.

## Configuration

The main configuration options are set in the Eagle_extractor.ipynb notebook. These include:

- Image processing options (e.g., resizing, cropping)
- Caption generation options
- Augmentation settings

## Dependencies

- PIL (Pillow)
- OpenCV (cv2)
- torch
- transformers
- tqdm
- numpy

Ensure all dependencies are installed before using these utilities.

## Note

This is a personal repository for working with the Eagle API. Use and modify as needed for your specific use case.
