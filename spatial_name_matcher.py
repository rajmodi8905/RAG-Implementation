# spatial_name_matcher.py

import fitz  # PyMuPDF
import os
import re
from collections import defaultdict
import json
import cv2
import numpy as np
from PIL import Image
import io

def extract_names_and_image_positions(pdf_path):
    """
    Extract names and their positions, along with image positions from the PDF.
    """
    pdf_document = fitz.open(pdf_path)
    
    page_data = {}
    
    print(f"Analyzing PDF: {pdf_path}")
    print(f"Total pages: {pdf_document.page_count}")
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        
        # Get text with position information
        text_dict = page.get_text("dict")
        
        # Get images with position information
        image_list = page.get_images(full=True)
        
        # Extract text blocks with positions
        text_blocks = []
        for block in text_dict["blocks"]:
            if "lines" in block:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            bbox = span["bbox"]  # (x0, y0, x1, y1)
                            text_blocks.append({
                                "text": text,
                                "bbox": bbox,
                                "x": (bbox[0] + bbox[2]) / 2,  # center x
                                "y": (bbox[1] + bbox[3]) / 2   # center y
                            })
        
        # Extract image positions
        image_positions = []
        for img_index, img in enumerate(image_list):
            # Get image rectangle
            img_rects = page.get_image_rects(img[0])
            if img_rects:
                for rect in img_rects:
                    bbox = rect  # fitz.Rect object
                    image_positions.append({
                        "img_index": img_index,
                        "xref": img[0],
                        "bbox": (bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                        "x": (bbox.x0 + bbox.x1) / 2,  # center x
                        "y": (bbox.y0 + bbox.y1) / 2   # center y
                    })
        
        if text_blocks or image_positions:
            page_data[page_num + 1] = {
                "text_blocks": text_blocks,
                "image_positions": image_positions
            }
            
            print(f"Page {page_num + 1}: {len(text_blocks)} text blocks, {len(image_positions)} images")
    
    pdf_document.close()
    return page_data

def extract_person_names_from_blocks(text_blocks):
    """
    Extract person names from text blocks.
    """
    names = []
    
    for block in text_blocks:
        text = block["text"]
        
        # Skip non-name text
        if skip_text(text):
            continue
            
        # Extract names from the text
        potential_names = find_names_in_text(text)
        
        for name in potential_names:
            if is_valid_person_name(name):
                cleaned_name = clean_name(name)
                if cleaned_name:
                    names.append({
                        "name": cleaned_name,
                        "bbox": block["bbox"],
                        "x": block["x"],
                        "y": block["y"]
                    })
    
    return names

def skip_text(text):
    """
    Check if text should be skipped (contains non-name information).
    """
    skip_patterns = [
        r'\b\d{4,}\b',  # Numbers with 4+ digits
        r'\bRoll\s*No',
        r'\bID\s*No',
        r'\b(?:Street|Road|Avenue|City|State|Country)\b',
        r'\b(?:Mumbai|Delhi|Bangalore|Chennai|Kolkata|Hyderabad|Pune|Ahmedabad)\b',
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Dates
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b',
        r'\b(?:Semester|Course|Subject|Grade|Marks|Department|College|University)\b',
        r'\b(?:Page|Total|Sum|Average|Report)\b',
        r'^\d+$',  # Pure numbers
        r'^[A-Z]{2,}$',  # All caps (likely abbreviations)
        r'\b(?:PE|PT|MANDATORY|COMPULSORY)\b',  # Common non-name terms
    ]
    
    for pattern in skip_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    # Skip very short text
    if len(text.strip()) < 3:
        return True
        
    return False

def find_names_in_text(text):
    """
    Find potential person names in text.
    """
    names = []
    
    # Clean the text
    text = text.strip()
    
    # Patterns for different name formats
    name_patterns = [
        # Standard format: First Last (or First Middle Last)
        r'\b([A-Z][a-z]{1,15})\s+([A-Z][a-z]{1,15})(?:\s+([A-Z][a-z]{1,15}))?\b',
        
        # With titles: Dr./Mr./Ms. First Last
        r'\b(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Prof\.?)\s+([A-Z][a-z]{1,15})\s+([A-Z][a-z]{1,15})\b',
        
        # Name with initials
        r'\b([A-Z][a-z]{1,15})\s+([A-Z]\.?\s*[A-Z]\.?)\b',
        r'\b([A-Z]\.?\s*[A-Z]\.?\s+[A-Z][a-z]{2,15})\b',
    ]
    
    for pattern in name_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            groups = [g for g in match.groups() if g]
            if len(groups) >= 2:
                name = ' '.join(groups)
                names.append(name.strip())
            elif len(groups) == 1:
                # Single group, check if it looks like a name
                if len(groups[0].split()) >= 2:
                    names.append(groups[0].strip())
    
    # If no pattern matches, check if the whole text looks like a name
    if not names:
        words = text.split()
        if 2 <= len(words) <= 4:  # Reasonable name length
            if all(word[0].isupper() and word[1:].islower() for word in words if word.isalpha()):
                names.append(text)
    
    return names

def is_valid_person_name(name):
    """
    Check if a name is likely to be a person's name.
    """
    if not name or len(name) < 3:
        return False
    
    words = name.split()
    if len(words) < 2:
        return False
    
    # Common false positives
    false_positives = {
        'Page Number', 'Roll Number', 'Student ID', 'Date Time', 'Class Room',
        'New York', 'Los Angeles', 'Computer Science', 'Information Technology',
        'Machine Learning', 'Artificial Intelligence', 'Software Engineering',
        'United States', 'Great Britain', 'Tamil Nadu', 'Andhra Pradesh',
        'Madhya Pradesh', 'Uttar Pradesh', 'West Bengal', 'Himachal Pradesh'
    }
    
    if name in false_positives:
        return False
    
    # Check individual words
    for word in words:
        if word.lower() in ['page', 'roll', 'number', 'id', 'date', 'time', 'class', 'grade', 'mandatory', 'compulsory']:
            return False
    
    # All words should be proper case
    for word in words:
        if not word[0].isupper() or not word[1:].islower():
            if not re.match(r'^[A-Z]\.?$', word):  # Allow initials
                return False
    
    return True

def clean_name(name):
    """
    Clean and standardize a name.
    """
    # Remove titles
    name = re.sub(r'^(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Prof\.?)\s+', '', name, flags=re.IGNORECASE)
    
    # Remove extra spaces
    name = ' '.join(name.split())
    
    # Capitalize properly
    words = []
    for word in name.split():
        if '.' in word:  # Handle initials
            words.append(word.upper())
        else:
            words.append(word.capitalize())
    
    return ' '.join(words)

def find_closest_image_to_name(name_info, image_positions, max_distance=200):
    """
    Find the image closest to a name, within a reasonable distance.
    """
    name_x, name_y = name_info["x"], name_info["y"]
    
    closest_image = None
    min_distance = float('inf')
    
    for img_pos in image_positions:
        img_x, img_y = img_pos["x"], img_pos["y"]
        
        # Calculate distance
        distance = ((name_x - img_x) ** 2 + (name_y - img_y) ** 2) ** 0.5
        
        if distance < min_distance and distance <= max_distance:
            min_distance = distance
            closest_image = img_pos
    
    return closest_image, min_distance

def create_name_to_image_mapping(pdf_path):
    """
    Create a mapping between names and their corresponding images.
    """
    page_data = extract_names_and_image_positions(pdf_path)
    
    name_image_pairs = []
    
    for page_num, data in page_data.items():
        text_blocks = data["text_blocks"]
        image_positions = data["image_positions"]
        
        # Extract names from text blocks
        names = extract_person_names_from_blocks(text_blocks)
        
        print(f"\nPage {page_num}:")
        print(f"  Found {len(names)} names: {[n['name'] for n in names[:5]]}")
        print(f"  Found {len(image_positions)} images")
        
        # Match each name with the closest image
        for name_info in names:
            closest_image, distance = find_closest_image_to_name(name_info, image_positions)
            
            if closest_image:
                name_image_pairs.append({
                    "name": name_info["name"],
                    "page": page_num,
                    "img_index": closest_image["img_index"],
                    "img_xref": closest_image["xref"],
                    "distance": distance,
                    "name_pos": (name_info["x"], name_info["y"]),
                    "img_pos": (closest_image["x"], closest_image["y"])
                })
                
                print(f"    '{name_info['name']}' → Image {closest_image['img_index']} (distance: {distance:.1f})")
    
    return name_image_pairs

def rename_images_with_mapping(pdf_path):
    """
    Rename face images based on spatial mapping with names in PDF.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        return
    
    # Get current face images
    test_images_dir = 'test_images'
    if not os.path.exists(test_images_dir):
        print(f"Error: '{test_images_dir}' folder not found.")
        return
    
    face_images = sorted([f for f in os.listdir(test_images_dir) 
                         if f.endswith(('png', 'jpg', 'jpeg'))])
    
    print(f"Found {len(face_images)} face images")
    
    # Create name-to-image mapping
    name_image_pairs = create_name_to_image_mapping(pdf_path)
    
    if not name_image_pairs:
        print("No name-image pairs found.")
        return
    
    print(f"\nFound {len(name_image_pairs)} name-image pairs")
    
    # Create a mapping from extracted image names to person names
    pdf_document = fitz.open(pdf_path)
    
    # Track which images correspond to which extracted files
    image_counter = 0
    renamed_count = 0
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            # Skip small images (likely not face photos)
            base_image = pdf_document.extract_image(img[0])
            if base_image["width"] < 100 or base_image["height"] < 100:
                continue
            
            # Find if this image has a corresponding name
            corresponding_name = None
            for pair in name_image_pairs:
                if (pair["page"] == page_num + 1 and 
                    pair["img_xref"] == img[0]):
                    corresponding_name = pair["name"]
                    break
            
            # Find the corresponding extracted image file
            current_image_file = f"extracted_image_{image_counter:03d}.{base_image['ext']}"
            current_image_path = os.path.join(test_images_dir, current_image_file)
            
            # Also check for files that might have been renamed already
            if not os.path.exists(current_image_path):
                # Look for any file that might correspond to this image
                possible_files = [f for f in face_images if f.startswith(f"extracted_image_{image_counter:03d}")]
                if possible_files:
                    current_image_file = possible_files[0]
                    current_image_path = os.path.join(test_images_dir, current_image_file)
            
            if corresponding_name and os.path.exists(current_image_path):
                # Create safe filename
                safe_name = re.sub(r'[^\w\s-]', '', corresponding_name)
                safe_name = re.sub(r'\s+', '_', safe_name)
                
                extension = current_image_file.split('.')[-1]
                new_filename = f"{safe_name}.{extension}"
                new_path = os.path.join(test_images_dir, new_filename)
                
                # Handle duplicates
                counter = 1
                while os.path.exists(new_path):
                    new_filename = f"{safe_name}_{counter}.{extension}"
                    new_path = os.path.join(test_images_dir, new_filename)
                    counter += 1
                
                try:
                    os.rename(current_image_path, new_path)
                    print(f"✓ {current_image_file} → {new_filename}")
                    renamed_count += 1
                except Exception as e:
                    print(f"✗ Error renaming {current_image_file}: {e}")
            
            elif os.path.exists(current_image_path):
                print(f"⚠ No name found for {current_image_file}")
            
            image_counter += 1
    
    pdf_document.close()
    
    print(f"\n" + "="*60)
    print(f"COMPLETED: Renamed {renamed_count} images based on spatial mapping")
    print("="*60)

def main():
    print("Spatial Name-Image Matcher")
    print("=" * 50)
    print("This tool matches names with their corresponding photos based on position in PDF.")
    print()
    
    pdf_path = input("Enter the path to your PDF file: ").strip()
    
    rename_images_with_mapping(pdf_path)

if __name__ == "__main__":
    main()