"""
Script to download the Sherlock Holmes book from Project Gutenberg
"""
import requests
import os

def download_book():
    """Download the Sherlock Holmes book from Project Gutenberg"""
    url = "https://www.gutenberg.org/files/1661/1661-0.txt"
    
    print("Downloading 'The Adventures of Sherlock Holmes' by Arthur Conan Doyle...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the book content to book.txt
        with open('book.txt', 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Book downloaded successfully! Size: {len(response.text)} characters")
        print("Saved as 'book.txt'")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading book: {e}")
        print("Please download the book manually from:")
        print("https://www.gutenberg.org/files/1661/1661-0.txt")
        print("and save it as 'book.txt'")

if __name__ == "__main__":
    download_book()