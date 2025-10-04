# extract_images_from_pdf.py

import fitz  # PyMuPDF
import os
from PIL import Image
import io
import cv2
import numpy as np

def extract_images_from_pdf(pdf_path, output_folder='test_images'):
    """
    Extract all images from a PDF file and save them to the output folder.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    
    image_count = 0
    extracted_images = []
    
    print(f"Processing PDF: {pdf_path}")
    print(f"Number of pages: {pdf_document.page_count}")
    
    # Iterate through each page
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        image_list = page.get_images(full=True)
        
        print(f"Page {page_num + 1}: Found {len(image_list)} images")
        
        # Extract each image from the page
        for img_index, img in enumerate(image_list):
            xref = img[0]  # Get image reference
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Convert bytes to PIL Image
            try:
                image = Image.open(io.BytesIO(image_bytes))
                
                # Skip very small images (likely not photos of people)
                if image.width < 100 or image.height < 100:
                    continue
                
                # Save the image
                image_filename = f"extracted_image_{image_count:03d}.{image_ext}"
                image_path = os.path.join(output_folder, image_filename)
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image.save(image_path)
                extracted_images.append(image_path)
                image_count += 1
                
                print(f"  Saved: {image_filename} ({image.width}x{image.height})")
                
            except Exception as e:
                print(f"  Error processing image {img_index}: {e}")
    
    pdf_document.close()
    
    print(f"\nExtraction complete!")
    print(f"Total images extracted: {image_count}")
    print(f"Images saved to: {output_folder}")
    
    return extracted_images

def has_face(image_path):
    """
    Simple face detection to check if an image likely contains a person.
    This is a basic implementation - you might want to use more sophisticated methods.
    """
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load OpenCV's pre-trained face detection classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Return True if at least one face is detected
        return len(faces) > 0
    
    except Exception as e:
        print(f"Error in face detection for {image_path}: {e}")
        return False

def filter_images_with_faces(image_paths):
    """
    Filter the extracted images to keep only those that likely contain faces.
    """
    face_images = []
    
    print("\nFiltering images for faces...")
    
    for image_path in image_paths:
        if has_face(image_path):
            face_images.append(image_path)
            print(f"  ✓ Face detected: {os.path.basename(image_path)}")
        else:
            # Remove images without faces
            try:
                os.remove(image_path)
                print(f"  ✗ No face, removed: {os.path.basename(image_path)}")
            except:
                pass
    
    print(f"\nKept {len(face_images)} images with faces")
    return face_images

def main():
    # You'll need to specify the path to your PDF file
    pdf_path = input("Enter the path to your PDF file: ").strip()
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        return
    
    # Extract all images from the PDF
    extracted_images = extract_images_from_pdf(pdf_path)
    
    if not extracted_images:
        print("No images were extracted from the PDF.")
        return
    
    # Filter for images with faces
    face_images = filter_images_with_faces(extracted_images)
    
    if face_images:
        print(f"\nSuccess! {len(face_images)} images with faces saved to 'test_images' folder.")
        print("\nNext steps:")
        print("1. Review the extracted images in the 'test_images' folder")
        print("2. Manually rename them based on the people's names if needed")
        print("3. Run the image_matcher.py script to test similarity matching")
    else:
        print("No faces were detected in the extracted images.")

if __name__ == "__main__":
    main()