import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

def setup_ocr_logging():
    """Setup logging for OCR processing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ocr_processing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def preprocess_image(image):
    """
    Preprocess image to improve OCR accuracy
    """
    # Convert PIL image to OpenCV format
    open_cv_image = np.array(image)
    
    # Convert RGB to BGR (OpenCV format)
    if len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 3:
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    
    # Apply noise removal
    denoised = cv2.medianBlur(gray, 3)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Optional: Apply morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return processed

def extract_text_from_image(image, config='--psm 6 -c preserve_interword_spaces=1'):
    """
    Extract text from a single image using Tesseract OCR
    """
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Convert back to PIL Image for pytesseract
        pil_image = Image.fromarray(processed_image)
        
        # Extract text using Tesseract
        text = pytesseract.image_to_string(pil_image, config=config, lang='eng')
        
        return text.strip()
    
    except Exception as e:
        logging.error(f"Error in OCR processing: {str(e)}")
        return ""

def process_single_pdf(pdf_path, output_dir, dpi=300):
    """
    Process a single PDF file: convert to images and perform OCR
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Get the base filename without extension
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}.txt")
        
        # Skip if output file already exists
        if os.path.exists(output_file):
            logger.info(f"Output file already exists, skipping: {output_file}")
            return True, pdf_path, output_file
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=dpi, first_page=None, last_page=None)
        
        if not images:
            logger.warning(f"No images extracted from PDF: {pdf_path}")
            return False, pdf_path, "No images extracted"
        
        logger.info(f"Converted {len(images)} pages to images")
        
        all_text = []
        
        # Process each page with OCR
        for page_num, image in enumerate(images, 1):
            logger.info(f"Processing page {page_num}/{len(images)}")
            
            # Extract text from the image
            page_text = extract_text_from_image(image)
            
            if page_text:
                all_text.append(f"--- Page {page_num} ---\n{page_text}\n")
            else:
                logger.warning(f"No text extracted from page {page_num}")
                all_text.append(f"--- Page {page_num} ---\n[No text extracted]\n")
        
        # Combine all text
        full_text = "\n".join(all_text)
        
        if not full_text.strip():
            logger.warning(f"No text extracted from PDF: {pdf_path}")
            return False, pdf_path, "No text extracted"
        
        # Save to text file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        logger.info(f"Successfully processed and saved: {output_file}")
        return True, pdf_path, output_file
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        return False, pdf_path, str(e)

def ocr_pdfs_to_text(input_folder, output_folder, dpi=300, max_workers=None):
    """
    Main function to process all PDFs in a folder (including subfolders) using OCR
    and save as text files.
    
    Args:
        input_folder (str): Path to folder containing PDF files and subfolders
        output_folder (str): Path to folder where text files will be saved
        dpi (int): DPI for PDF to image conversion (higher = better quality but slower)
        max_workers (int): Number of parallel workers (None = auto-detect)
    
    Returns:
        dict: Processing statistics
    """
    
    # Setup logging
    logger = setup_ocr_logging()
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all PDF files
    pdf_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    if not pdf_files:
        logger.warning("No PDF files found in the specified folder")
        return {"total": 0, "successful": 0, "failed": 0, "errors": []}
    
    # Process statistics
    stats = {
        "total": len(pdf_files),
        "successful": 0,
        "failed": 0,
        "errors": []
    }
    
    # Process PDFs (optionally in parallel)
    if max_workers and max_workers > 1:
        logger.info(f"Processing {len(pdf_files)} PDFs with {max_workers} parallel workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pdf = {
                executor.submit(process_single_pdf, pdf_path, output_folder, dpi): pdf_path 
                for pdf_path in pdf_files
            }
            
            # Process completed tasks
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    success, processed_pdf, result = future.result()
                    if success:
                        stats["successful"] += 1
                        logger.info(f"✅ Success: {processed_pdf}")
                    else:
                        stats["failed"] += 1
                        stats["errors"].append(f"{processed_pdf}: {result}")
                        logger.error(f"❌ Failed: {processed_pdf} - {result}")
                except Exception as e:
                    stats["failed"] += 1
                    stats["errors"].append(f"{pdf_path}: {str(e)}")
                    logger.error(f"❌ Exception: {pdf_path} - {str(e)}")
    else:
        # Process sequentially
        logger.info(f"Processing {len(pdf_files)} PDFs sequentially")
        for pdf_path in pdf_files:
            success, processed_pdf, result = process_single_pdf(pdf_path, output_folder, dpi)
            if success:
                stats["successful"] += 1
                logger.info(f"✅ Success: {processed_pdf}")
            else:
                stats["failed"] += 1
                stats["errors"].append(f"{processed_pdf}: {result}")
                logger.error(f"❌ Failed: {processed_pdf} - {result}")
    
    # Print summary
    logger.info(f"\n📊 PROCESSING SUMMARY:")
    logger.info(f"📂 Total PDFs processed: {stats['total']}")
    logger.info(f"✅ Successful: {stats['successful']}")
    logger.info(f"❌ Failed: {stats['failed']}")
    logger.info(f"📁 Output folder: {output_folder}")
    
    if stats['errors']:
        logger.info(f"📋 Errors encountered:")
        for error in stats['errors']:
            logger.info(f"   - {error}")
    
    return stats

# Additional utility function for batch processing with different configurations
def batch_ocr_processing(input_folder, output_base_folder, dpi_list=[200, 300, 400]):
    """
    Process PDFs with different DPI settings for quality comparison
    """
    results = {}
    
    for dpi in dpi_list:
        output_folder = os.path.join(output_base_folder, f"dpi_{dpi}")
        logger.info(f"\n🔧 Processing with DPI: {dpi}")
        
        stats = ocr_pdfs_to_text(input_folder, output_folder, dpi=dpi, max_workers=2)
        results[dpi] = stats
    
    return results

# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage
    input_dir = "data/ADULT AND CONTINUING EDUCATION"
    output_dir = "ocr_data_output"
    
    # Process all PDFs with OCR
    stats = ocr_pdfs_to_text(
        input_folder=input_dir,
        output_folder=output_dir,
        dpi=300,  # Good balance of quality and speed
        max_workers=1 # Process 4 PDFs in parallel
    )
    
    print(f"Processing complete! Success: {stats['successful']}/{stats['total']}")
    
    # Example 2: Batch processing with different DPI settings
    # batch_results = batch_ocr_processing(input_dir, output_dir)