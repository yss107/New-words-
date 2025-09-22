#!/usr/bin/env python3
"""
Text Preprocessing and Analysis Module for Next Word Prediction
"""
import re
import string
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

class TextPreprocessor:
    """Handles text cleaning, tokenization and basic analysis"""
    
    def __init__(self, min_word_length: int = 2):
        self.min_word_length = min_word_length
        self.vocabulary = set()
        self.word_to_id = {}
        self.id_to_word = {}
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove Project Gutenberg headers/footers if present
        lines = text.split('\n')
        start_idx = 0
        end_idx = len(lines)
        
        # Find actual story start (skip headers)
        for i, line in enumerate(lines):
            if 'to sherlock holmes' in line.lower() or 'sherlock holmes' in line.lower():
                start_idx = i
                break
        
        # Remove potential footer content
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() and not lines[i].lower().startswith('end of'):
                end_idx = i + 1
                break
        
        text = '\n'.join(lines[start_idx:end_idx])
        
        # Replace multiple whitespaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra punctuation but keep sentence-ending punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Split on whitespace and punctuation
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        
        # Filter by minimum length and remove empty strings
        words = [word.lower() for word in words if len(word) >= self.min_word_length]
        
        return words
    
    def build_vocabulary(self, words: List[str]) -> None:
        """Build vocabulary and word mappings"""
        self.vocabulary = set(words)
        
        # Create word to ID mappings
        unique_words = sorted(list(self.vocabulary))
        self.word_to_id = {word: i for i, word in enumerate(unique_words)}
        self.id_to_word = {i: word for word, i in self.word_to_id.items()}
    
    def get_word_frequencies(self, words: List[str]) -> Counter:
        """Get word frequency counts"""
        return Counter(words)
    
    def get_word_pairs(self, words: List[str]) -> Counter:
        """Get consecutive word pair frequencies"""
        pairs = []
        for i in range(len(words) - 1):
            pairs.append((words[i], words[i + 1]))
        return Counter(pairs)
    
    def analyze_text(self, words: List[str]) -> Dict:
        """Perform comprehensive text analysis"""
        word_freq = self.get_word_frequencies(words)
        word_pairs = self.get_word_pairs(words)
        
        analysis = {
            'total_words': len(words),
            'unique_words': len(self.vocabulary),
            'vocab_size': len(self.vocabulary),
            'most_common_words': word_freq.most_common(20),
            'least_common_words': word_freq.most_common()[-20:],
            'most_common_pairs': word_pairs.most_common(20),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0
        }
        
        return analysis

class DatasetGenerator:
    """Generate training datasets for neural network"""
    
    def __init__(self, sequence_length: int = 5):
        self.sequence_length = sequence_length
    
    def create_sequences(self, words: List[str]) -> Tuple[List[List[str]], List[str]]:
        """Create input-output sequences for training"""
        inputs = []
        outputs = []
        
        for i in range(len(words) - self.sequence_length):
            # Input: sequence of words
            input_seq = words[i:i + self.sequence_length]
            # Output: next word
            output_word = words[i + self.sequence_length]
            
            inputs.append(input_seq)
            outputs.append(output_word)
        
        return inputs, outputs
    
    def encode_sequences(self, sequences: List[List[str]], word_to_id: Dict[str, int]) -> List[List[int]]:
        """Encode word sequences to ID sequences"""
        encoded = []
        for seq in sequences:
            encoded_seq = [word_to_id.get(word, 0) for word in seq]
            encoded.append(encoded_seq)
        return encoded
    
    def encode_words(self, words: List[str], word_to_id: Dict[str, int]) -> List[int]:
        """Encode words to IDs"""
        return [word_to_id.get(word, 0) for word in words]

def main():
    """Demonstrate text preprocessing and analysis"""
    print("Loading and preprocessing text...")
    
    # Load text
    with open('book.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Preprocess
    preprocessor = TextPreprocessor()
    cleaned_text = preprocessor.clean_text(text)
    words = preprocessor.tokenize(cleaned_text)
    preprocessor.build_vocabulary(words)
    
    # Analyze
    analysis = preprocessor.analyze_text(words)
    
    print(f"\n=== TEXT ANALYSIS ===")
    print(f"Total words: {analysis['total_words']}")
    print(f"Unique words: {analysis['unique_words']}")
    print(f"Average word length: {analysis['avg_word_length']:.2f}")
    
    print(f"\n=== MOST COMMON WORDS ===")
    for word, count in analysis['most_common_words']:
        print(f"{word}: {count}")
    
    print(f"\n=== MOST COMMON WORD PAIRS ===")
    for (word1, word2), count in analysis['most_common_pairs']:
        print(f"'{word1} {word2}': {count}")
    
    # Generate datasets
    print(f"\n=== DATASET GENERATION ===")
    generator = DatasetGenerator(sequence_length=5)
    inputs, outputs = generator.create_sequences(words)
    
    print(f"Total training sequences: {len(inputs)}")
    print(f"Example input: {inputs[0]}")
    print(f"Example output: {outputs[0]}")
    
    # Show a few more examples
    print(f"\nFirst 5 training examples:")
    for i in range(5):
        print(f"Input: {' '.join(inputs[i])} -> Output: {outputs[i]}")

if __name__ == "__main__":
    main()