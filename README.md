# True/False Answer Evaluation System

This project uses **pytesseract** for OCR (Optical Character Recognition) and **Streamlit** for the web interface. 

## Prerequisites

1. **Install Tesseract OCR**
   - Windows: Download and install Tesseract from [this link](https://github.com/UB-Mannheim/tesseract/wiki).
   - Linux: You can install it via the terminal using `sudo apt install tesseract-ocr`.
   - Make sure the Tesseract executable is in your system's PATH.

2. **Install dependencies**:
   - Clone the repository:
     ```bash
     git clone https://github.com/yourusername/yourproject.git
     cd yourproject
     ```
   - Create a virtual environment (optional but recommended):
     ```bash
     python -m venv venv
     source venv/bin/activate   # On Windows use `venv\Scripts\activate`
     ```
   - Install the dependencies:
     ```bash
     pip install -r requirements.txt
     ```

## Running the Project

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
