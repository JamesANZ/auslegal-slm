#!/usr/bin/env python3
"""
Clean Legal Data Files

Strips metadata headers and cleans irrelevant content from legal document files.
This is done once to clean the source data files, rather than cleaning during each preprocessing run.
"""

import os
import re
from pathlib import Path
from typing import List


def clean_legal_text(text: str) -> str:
    """
    Clean and normalize legal text content.
    
    Removes:
    - Metadata headers (URL, Scraped date, separators)
    - Navigation/UI elements
    - Excessive whitespace
    - Common non-legal text patterns
    
    Args:
        text: Raw text content from file
        
    Returns:
        Cleaned legal text
    """
    lines = text.split('\n')
    
    # Find and remove metadata header
    # Format: URL: ...\nScraped: ...\n================================================================================\n\n
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('=' * 20):  # Separator line
            start_idx = i + 1
            break
        # Also check for lines that are clearly metadata
        if line.startswith('URL:') or line.startswith('Scraped:'):
            start_idx = i + 1
    
    # Extract content after metadata
    content_lines = lines[start_idx:] if start_idx > 0 else lines
    
    # Remove common navigation/UI text patterns
    skip_patterns = [
        r'^Cases & Legislation',
        r'^Journals & Scholarship',
        r'^Communities',
        r'^New Zealand',
        r'^Specific Year',
        r'^Most Recent',
        r'^Most Accessed',
        r'^Print \(',
        r'^PDF format',
        r'^LawCite records',
        r'^NoteUp references',
        r'^Join the discussion',
        r'^Tweet this page',
        r'^Follow @',
        r'^Subscribe with RSS',
        r'^Subscribe to database feed',
        r'^This database contains',
        r'^number of documents',
        r'^accesses in the last',
        r'^most recent document',
    ]
    
    filtered_lines = []
    for line in content_lines:
        line = line.strip()
        if not line:
            # Keep single blank lines, but remove excessive whitespace
            if filtered_lines and filtered_lines[-1]:  # Only add blank if previous line wasn't blank
                filtered_lines.append('')
            continue
        
        # Skip lines matching skip patterns
        should_skip = any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns)
        if should_skip:
            continue
        
        # Skip very short lines that are likely navigation
        if len(line) < 5:
            continue
        
        # Skip lines that are just dates or numbers
        if re.match(r'^\d{1,2}\s+\w+\s+\d{4}$', line):  # Date patterns
            continue
        
        filtered_lines.append(line)
    
    # Join lines and clean up whitespace
    cleaned_text = '\n'.join(filtered_lines)
    
    # Remove excessive whitespace
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  # Max 2 consecutive newlines
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)  # Multiple spaces to single
    
    return cleaned_text.strip()


def clean_data_file(file_path: Path) -> tuple[bool, bool]:
    """
    Clean a single data file in place.
    
    Args:
        file_path: Path to the file to clean
        
    Returns:
        Tuple of (success: bool, was_error: bool)
        - success: True if file was cleaned successfully, False otherwise
        - was_error: True if an exception occurred, False if just skipped (too short)
    """
    try:
        # Read original file with error handling for encoding issues
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            original_content = f.read()
        
        # Clean the content
        cleaned_content = clean_legal_text(original_content)
        
        # Skip if cleaned content is too short (likely an error or empty page)
        if len(cleaned_content) < 100:
            print(f"  Skipping {file_path.name}: cleaned content too short ({len(cleaned_content)} chars)")
            return False, False  # Not an error, just skipped
        
        # Write cleaned content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        return True, False  # Success, no error
        
    except Exception as e:
        print(f"  Error cleaning {file_path.name}: {e}")
        return False, True  # Failed due to error


def main():
    """Main cleaning function."""
    DATA_DIR = "data"
    
    print("=" * 80)
    print("Cleaning Legal Data Files")
    print("=" * 80)
    print(f"Processing files in {DATA_DIR}/...")
    print()
    
    data_path = Path(DATA_DIR)
    
    # Get all .txt files (excluding hidden files)
    txt_files = sorted([f for f in data_path.glob("*.txt") if not f.name.startswith('.')])
    
    if not txt_files:
        print(f"ERROR: No .txt files found in {DATA_DIR}/")
        return
    
    print(f"Found {len(txt_files)} files to clean")
    print()
    
    cleaned_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, file_path in enumerate(txt_files, 1):
        success, was_error = clean_data_file(file_path)
        if success:
            cleaned_count += 1
        elif was_error:
            error_count += 1
        else:
            skipped_count += 1
        
        if (i % 100 == 0) or (i == len(txt_files)):
            print(f"  Progress: {i}/{len(txt_files)} files processed ({cleaned_count} cleaned, {skipped_count} skipped, {error_count} errors)")
    
    print()
    print("=" * 80)
    print("Cleaning Complete!")
    print("=" * 80)
    print(f"  Total files: {len(txt_files)}")
    print(f"  Successfully cleaned: {cleaned_count}")
    print(f"  Skipped (too short/empty): {skipped_count}")
    print(f"  Errors: {error_count}")
    print()
    print("Data files have been cleaned. You can now run prepare_data.py")


if __name__ == "__main__":
    main()

