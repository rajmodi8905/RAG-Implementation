# extract_names_and_rename.py

import fitz  # PyMuPDF
import os
import re
from PIL import Image
import io
import cv2
import numpy as np
from collections import defaultdict
import json

def extract_text_from_pdf(pdf_path):
    """
    Extract all text from the PDF, page by page.
    """
    pdf_document = fitz.open(pdf_path)
    page_texts = {}
    
    print(f"Extracting text from PDF: {pdf_path}")
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text = page.get_text()
        page_texts[page_num + 1] = text  # 1-indexed page numbers
        
    pdf_document.close()
    return page_texts

def extract_names_from_text(text):
    """
    Extract potential names from text using various patterns.
    This is a basic implementation - you might need to adjust patterns based on your PDF format.
    """
    names = set()
    
    # Common name patterns
    patterns = [
        # Full names (First Last, First Middle Last)
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b',
        # Names with titles (Dr. John Smith, Mr. Jane Doe)
        r'\b(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Prof\.?)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
        # Names in format "LastName, FirstName"
        r'\b[A-Z][a-z]+,\s+[A-Z][a-z]+\b',
        # Names after common prefixes
        r'(?:Name:|Student:|Employee:|Person:)\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]  # Extract from group
            
            # Clean up the name
            name = re.sub(r'^(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Prof\.?)\s+', '', match.strip())
            name = re.sub(r'[,:]+', '', name)  # Remove punctuation
            
            # Filter out common false positives
            false_positives = {'Page', 'Date', 'Time', 'Year', 'Class', 'Grade', 'School', 'University', 'College'}
            if not any(fp in name for fp in false_positives) and len(name) > 3:
                names.add(name.strip())
    
    return list(names)

def find_names_near_images(pdf_path):
    """
    Extract names and try to associate them with image positions in the PDF.
    """
    pdf_document = fitz.open(pdf_path)
    
    names_by_page = {}
    images_by_page = {}
    
    print("Analyzing PDF structure...")
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        
        # Get text with position information
        text_dict = page.get_text("dict")
        page_text = page.get_text()
        
        # Get images on this page
        image_list = page.get_images(full=True)
        
        # Extract names from page text
        names = extract_names_from_text(page_text)
        
        if names or image_list:
            names_by_page[page_num + 1] = names
            images_by_page[page_num + 1] = len(image_list)
            
            if names:
                print(f"Page {page_num + 1}: Found {len(names)} names and {len(image_list)} images")
                for name in names[:5]:  # Show first 5 names
                    print(f"  - {name}")
                if len(names) > 5:
                    print(f"  ... and {len(names) - 5} more names")
    
    pdf_document.close()
    
    return names_by_page, images_by_page

def create_name_mapping():
    """
    Create a mapping between extracted images and names.
    This is a simplified approach - in practice, you might need more sophisticated matching.
    """
    # Get list of current face images
    test_images_dir = 'test_images'
    face_images = sorted([f for f in os.listdir(test_images_dir) 
                         if f.endswith(('png', 'jpg', 'jpeg'))])
    
    print(f"\nFound {len(face_images)} face images to rename")
    
    # You'll need to specify the path to your PDF
    pdf_path = input("Enter the path to your PDF file: ").strip()
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        return None
    
    # Extract names from PDF
    names_by_page, images_by_page = find_names_near_images(pdf_path)
    
    # Collect all unique names
    all_names = set()
    for page_names in names_by_page.values():
        all_names.update(page_names)
    
    all_names = sorted(list(all_names))
    print(f"\nFound {len(all_names)} unique names in the PDF:")
    for i, name in enumerate(all_names, 1):
        print(f"{i:3d}. {name}")
    
    return all_names, face_images, names_by_page, images_by_page

def interactive_naming():
    """
    Interactive mode to manually associate names with face images.
    """
    result = create_name_mapping()
    if not result:
        return
    
    all_names, face_images, names_by_page, images_by_page = result
    
    if not all_names:
        print("No names found in the PDF. Please check the PDF format.")
        return
    
    print(f"\n" + "="*60)
    print("INTERACTIVE NAMING MODE")
    print("="*60)
    print("We'll go through each face image and you can assign a name to it.")
    print("You can:")
    print("- Enter a number to select from the extracted names")
    print("- Type a custom name directly")
    print("- Type 'skip' to skip this image")
    print("- Type 'quit' to exit")
    print("="*60)
    
    renamed_count = 0
    
    for i, image_file in enumerate(face_images[:20], 1):  # Limit to first 20 for demo
        print(f"\nImage {i}/{min(20, len(face_images))}: {image_file}")
        
        # Show available names
        print("Available names:")
        for j, name in enumerate(all_names[:15], 1):  # Show first 15 names
            print(f"{j:2d}. {name}")
        if len(all_names) > 15:
            print(f"    ... and {len(all_names) - 15} more names")
        
        while True:
            choice = input(f"\nEnter number (1-{len(all_names)}), custom name, 'skip', or 'quit': ").strip()
            
            if choice.lower() == 'quit':
                print(f"Exiting. Renamed {renamed_count} images.")
                return
            
            if choice.lower() == 'skip':
                print("Skipped.")
                break
            
            # Check if it's a number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(all_names):
                    selected_name = all_names[idx]
                    new_filename = f"{selected_name.replace(' ', '_')}.{image_file.split('.')[-1]}"
                    
                    # Rename the file
                    old_path = os.path.join('test_images', image_file)
                    new_path = os.path.join('test_images', new_filename)
                    
                    try:
                        os.rename(old_path, new_path)
                        print(f"✓ Renamed to: {new_filename}")
                        renamed_count += 1
                        break
                    except Exception as e:
                        print(f"Error renaming file: {e}")
                else:
                    print("Invalid number. Please try again.")
            else:
                # Custom name
                if choice and len(choice) > 1:
                    # Clean the name for filename
                    clean_name = re.sub(r'[^\w\s-]', '', choice).strip()
                    clean_name = re.sub(r'\s+', '_', clean_name)
                    
                    if clean_name:
                        new_filename = f"{clean_name}.{image_file.split('.')[-1]}"
                        old_path = os.path.join('test_images', image_file)
                        new_path = os.path.join('test_images', new_filename)
                        
                        try:
                            os.rename(old_path, new_path)
                            print(f"✓ Renamed to: {new_filename}")
                            renamed_count += 1
                            break
                        except Exception as e:
                            print(f"Error renaming file: {e}")
                    else:
                        print("Invalid name. Please try again.")
                else:
                    print("Please enter a valid name or number.")
    
    print(f"\n" + "="*60)
    print(f"COMPLETED: Renamed {renamed_count} images")
    print("="*60)
    
    # Show some renamed files
    renamed_files = [f for f in os.listdir('test_images') 
                    if not f.startswith('extracted_image_')]
    if renamed_files:
        print(f"Sample renamed files:")
        for f in renamed_files[:10]:
            print(f"  - {f}")

def main():
    print("Face Image Naming Tool")
    print("=====================")
    print("This tool will help you rename face images based on names found in the PDF.")
    print()
    
    interactive_naming()

if __name__ == "__main__":
    main()