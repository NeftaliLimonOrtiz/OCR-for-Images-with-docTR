# OCR-for-Images-with-docTR
This project implements an OCR preprocessing pipeline for construction site reports in PDF format.  
It uses **docTR** for text recognition and **OpenCV** for detecting table regions, reconstructing both free text and tabular content into txt files.

## Overview

The pipeline performs the following steps:

1. Loads PDF reports.
2. Applies OCR using docTR.
3. Detects table regions using horizontal and vertical line extraction.
4. Assigns OCR words to table cells.
5. Separates table content from free text.
6. Exports the content as `.txt` files.

## Installation
Install dependencies:

```bash
pip install -r requirements.txt
```
## Usage

Edit the following paths in main.py:
```bash
PDF_PATH = "path/to/pdf/or/folder"
OUTPUT_DIR = "path/to/output/folder"
```
Then run:

```bash
python main.py
```