#!/usr/bin/env python3
"""
Download The Adventures of Sherlock Holmes from Project Gutenberg
"""
import requests
import os

def download_book():
    """Download the book from Project Gutenberg if not already present"""
    book_path = "book.txt"
    
    if os.path.exists(book_path):
        print(f"Book already exists at {book_path}")
        return book_path
    
    # URL for The Adventures of Sherlock Holmes from Project Gutenberg
    url = "https://www.gutenberg.org/files/1661/1661-0.txt"
    
    print("Downloading The Adventures of Sherlock Holmes...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(book_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Successfully downloaded book to {book_path}")
        return book_path
    
    except requests.RequestException as e:
        print(f"Error downloading book: {e}")
        return None

if __name__ == "__main__":
    download_book()