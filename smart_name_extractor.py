# smart_name_extractor.py

import fitz  # PyMuPDF
import os
import re
from collections import defaultdict
import json

def extract_person_names_from_pdf(pdf_path):
    """
    Extract only person names from PDF, filtering out roll numbers, locations, etc.
    """
    pdf_document = fitz.open(pdf_path)
    
    all_names = []
    names_by_page = {}
    
    print(f"Extracting names from PDF: {pdf_path}")
    print(f"Total pages: {pdf_document.page_count}")
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text = page.get_text()
        
        # Extract names from this page
        page_names = extract_names_from_text(text)
        
        if page_names:
            names_by_page[page_num + 1] = page_names
            all_names.extend(page_names)
            print(f"Page {page_num + 1}: Found {len(page_names)} names")
    
    pdf_document.close()
    
    # Remove duplicates while preserving order
    unique_names = []
    seen = set()
    for name in all_names:
        if name not in seen:
            unique_names.append(name)
            seen.add(name)
    
    return unique_names, names_by_page

def extract_names_from_text(text):
    """
    Extract person names from text, filtering out non-person entities.
    """
    names = []
    
    # Split text into lines for better processing
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip lines that clearly contain non-names
        if skip_line(line):
            continue
            
        # Extract potential names from the line
        potential_names = find_names_in_line(line)
        names.extend(potential_names)
    
    # Filter and clean names
    filtered_names = []
    for name in names:
        if is_valid_person_name(name):
            cleaned_name = clean_name(name)
            if cleaned_name and cleaned_name not in filtered_names:
                filtered_names.append(cleaned_name)
    
    return filtered_names

def skip_line(line):
    """
    Check if a line should be skipped (contains non-name information).
    """
    skip_patterns = [
        # Roll numbers and IDs
        r'\b\d{4,}\b',  # 4+ digit numbers (roll numbers, IDs)
        r'\bRoll\s*No',
        r'\bID\s*No',
        r'\bStudent\s*ID',
        
        # Locations and addresses
        r'\b(?:Street|Road|Avenue|Lane|City|State|Country|Address)\b',
        r'\b(?:Mumbai|Delhi|Bangalore|Chennai|Kolkata|Hyderabad|Pune|Ahmedabad)\b',
        
        # Dates and times
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b',
        
        # Academic terms
        r'\b(?:Semester|Course|Subject|Grade|Marks|Score|Percentage)\b',
        r'\b(?:B\.Tech|M\.Tech|B\.Sc|M\.Sc|B\.A|M\.A|PhD|MBA)\b',
        
        # Common non-name words
        r'\b(?:Department|Faculty|College|University|Institute|School)\b',
        r'\b(?:Page|Total|Sum|Average|Result|Report)\b',
    ]
    
    for pattern in skip_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    
    return False

def find_names_in_line(line):
    """
    Find potential person names in a line of text.
    """
    names = []
    
    # Patterns for different name formats
    name_patterns = [
        # Standard format: First Last (or First Middle Last)
        r'\b([A-Z][a-z]{1,15})\s+([A-Z][a-z]{1,15})(?:\s+([A-Z][a-z]{1,15}))?\b',
        
        # With titles: Dr./Mr./Ms. First Last
        r'\b(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Prof\.?)\s+([A-Z][a-z]{1,15})\s+([A-Z][a-z]{1,15})\b',
        
        # Comma separated: Last, First
        r'\b([A-Z][a-z]{1,15}),\s+([A-Z][a-z]{1,15})\b',
        
        # Name with initials: A. B. Smith or A.B. Smith
        r'\b([A-Z]\.?\s*[A-Z]\.?\s+[A-Z][a-z]{2,15})\b',
        r'\b([A-Z][a-z]{1,15}\s+[A-Z]\.?\s*[A-Z]\.?)\b',
    ]
    
    for pattern in name_patterns:
        matches = re.finditer(pattern, line)
        for match in matches:
            if pattern.endswith(r'Last, First\b'):
                # Handle "Last, First" format
                last, first = match.groups()
                name = f"{first} {last}"
            elif 'Dr\\.|Mr\\.' in pattern:
                # Handle titled names
                first, last = match.groups()
                name = f"{first} {last}"
            else:
                # Handle other formats
                groups = [g for g in match.groups() if g]
                if len(groups) >= 2:
                    name = ' '.join(groups)
                else:
                    name = groups[0] if groups else ''
            
            if name:
                names.append(name.strip())
    
    return names

def is_valid_person_name(name):
    """
    Check if a name is likely to be a person's name.
    """
    if not name or len(name) < 3:
        return False
    
    # Split into words
    words = name.split()
    if len(words) < 2:
        return False
    
    # Common false positives to exclude
    false_positives = {
        'Page Number', 'Roll Number', 'Student ID', 'Date Time', 'Class Room',
        'First Name', 'Last Name', 'Full Name', 'User Name', 'File Name',
        'Test Case', 'Data Base', 'Web Site', 'Email ID', 'Phone Number',
        'Credit Card', 'Bank Account', 'Post Office', 'High School',
        'New York', 'Los Angeles', 'San Francisco', 'Las Vegas',
        'United States', 'Great Britain', 'South Africa', 'New Zealand',
        'Computer Science', 'Information Technology', 'Data Science',
        'Machine Learning', 'Artificial Intelligence', 'Software Engineering'
    }
    
    if name in false_positives:
        return False
    
    # Check if any word is a common false positive
    for word in words:
        if word.lower() in ['page', 'roll', 'number', 'id', 'date', 'time', 'class', 'grade']:
            return False
    
    # All words should start with capital letter (proper names)
    for word in words:
        if not word[0].isupper():
            return False
        
        # Word should be mostly alphabetic
        if not re.match(r'^[A-Za-z\.]+$', word):
            return False
    
    # Name shouldn't be too long (likely not a person's name)
    if len(name) > 50:
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
        if '.' in word:  # Handle initials like "A."
            words.append(word.upper())
        else:
            words.append(word.capitalize())
    
    return ' '.join(words)

def auto_rename_images(pdf_path):
    """
    Automatically extract names and rename face images.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        return
    
    # Get list of current face images
    test_images_dir = 'test_images'
    if not os.path.exists(test_images_dir):
        print(f"Error: '{test_images_dir}' folder not found.")
        return
    
    face_images = sorted([f for f in os.listdir(test_images_dir) 
                         if f.endswith(('png', 'jpg', 'jpeg')) and f.startswith('extracted_image_')])
    
    if not face_images:
        print("No extracted face images found to rename.")
        return
    
    print(f"Found {len(face_images)} face images to rename")
    
    # Extract names from PDF
    names, names_by_page = extract_person_names_from_pdf(pdf_path)
    
    if not names:
        print("No valid person names found in the PDF.")
        return
    
    print(f"\nExtracted {len(names)} unique names:")
    for i, name in enumerate(names, 1):
        print(f"{i:3d}. {name}")
    
    # Simple strategy: assign names to images in order
    print(f"\nRenaming images...")
    renamed_count = 0
    
    for i, image_file in enumerate(face_images):
        if i < len(names):
            name = names[i]
            # Create safe filename
            safe_name = re.sub(r'[^\w\s-]', '', name)
            safe_name = re.sub(r'\s+', '_', safe_name)
            
            extension = image_file.split('.')[-1]
            new_filename = f"{safe_name}.{extension}"
            
            old_path = os.path.join(test_images_dir, image_file)
            new_path = os.path.join(test_images_dir, new_filename)
            
            # Handle duplicate names
            counter = 1
            while os.path.exists(new_path):
                new_filename = f"{safe_name}_{counter}.{extension}"
                new_path = os.path.join(test_images_dir, new_filename)
                counter += 1
            
            try:
                os.rename(old_path, new_path)
                print(f"✓ {image_file} → {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"✗ Error renaming {image_file}: {e}")
        else:
            print(f"⚠ No name available for {image_file}")
    
    print(f"\n" + "="*50)
    print(f"COMPLETED: Renamed {renamed_count} out of {len(face_images)} images")
    print("="*50)
    
    # Show renamed files
    renamed_files = [f for f in os.listdir(test_images_dir) 
                    if not f.startswith('extracted_image_')]
    if renamed_files:
        print(f"\nRenamed files (sample):")
        for f in sorted(renamed_files)[:10]:
            print(f"  - {f}")
        if len(renamed_files) > 10:
            print(f"  ... and {len(renamed_files) - 10} more files")

def main():
    print("Smart Name Extractor and Image Renamer")
    print("=" * 50)
    print("This tool extracts person names from PDF and renames face images.")
    print()
    
    # You'll need to specify the path to your PDF
    pdf_path = input("Enter the path to your PDF file: ").strip()
    
    auto_rename_images(pdf_path)

if __name__ == "__main__":
    main()