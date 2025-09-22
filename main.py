#!/usr/bin/env python3
"""
Main Next Word Prediction System
Complete pipeline for training and testing the model
"""
import os
import pickle
import numpy as np
from text_processor import TextPreprocessor, DatasetGenerator
from neural_network import NextWordPredictor, ModelTrainer

class NextWordPredictionSystem:
    """Complete next word prediction system"""
    
    def __init__(self, text_file: str = 'book.txt', sequence_length: int = 5):
        self.text_file = text_file
        self.sequence_length = sequence_length
        self.preprocessor = None
        self.generator = None
        self.predictor = None
        self.word_to_id = {}
        self.id_to_word = {}
        
    def load_and_process_data(self):
        """Load and preprocess the text data"""
        print("Loading and preprocessing text data...")
        
        # Load text
        with open(self.text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor()
        
        # Clean and tokenize
        cleaned_text = self.preprocessor.clean_text(text)
        words = self.preprocessor.tokenize(cleaned_text)
        
        # Build vocabulary
        self.preprocessor.build_vocabulary(words)
        self.word_to_id = self.preprocessor.word_to_id
        self.id_to_word = self.preprocessor.id_to_word
        
        # Analyze text
        analysis = self.preprocessor.analyze_text(words)
        self.print_analysis(analysis)
        
        return words
    
    def print_analysis(self, analysis):
        """Print text analysis results"""
        print(f"\n=== TEXT ANALYSIS ===")
        print(f"Total words: {analysis['total_words']}")
        print(f"Unique words: {analysis['unique_words']}")
        print(f"Average word length: {analysis['avg_word_length']:.2f}")
        
        print(f"\nTop 10 most common words:")
        for word, count in analysis['most_common_words'][:10]:
            print(f"  {word}: {count}")
        
        print(f"\nTop 10 most common word pairs:")
        for (word1, word2), count in analysis['most_common_pairs'][:10]:
            print(f"  '{word1} {word2}': {count}")
    
    def prepare_training_data(self, words):
        """Prepare training sequences"""
        print(f"\nPreparing training data with sequence length {self.sequence_length}...")
        
        # Initialize generator
        self.generator = DatasetGenerator(self.sequence_length)
        
        # Create sequences
        input_sequences, output_words = self.generator.create_sequences(words)
        
        # Encode to IDs
        encoded_inputs = self.generator.encode_sequences(input_sequences, self.word_to_id)
        encoded_outputs = self.generator.encode_words(output_words, self.word_to_id)
        
        print(f"Created {len(encoded_inputs)} training sequences")
        print(f"Vocabulary size: {len(self.word_to_id)}")
        
        return encoded_inputs, encoded_outputs
    
    def train_model(self, encoded_inputs, encoded_outputs, epochs=50, batch_size=32):
        """Train the neural network model"""
        print(f"\nTraining neural network model...")
        
        # Create trainer
        trainer = ModelTrainer(len(self.word_to_id), self.sequence_length)
        
        # Train model
        results = trainer.train_model(
            encoded_inputs, encoded_outputs, 
            epochs=epochs, batch_size=batch_size
        )
        
        self.predictor = results['predictor']
        return results
    
    def test_prediction(self, test_input: str, top_k: int = 5):
        """Test prediction with custom input"""
        if self.predictor is None:
            print("Model must be trained before prediction!")
            return
        
        print(f"\nTesting prediction for: '{test_input}'")
        
        # Tokenize input
        words = self.preprocessor.tokenize(test_input)
        
        # Take last sequence_length words
        if len(words) < self.sequence_length:
            print(f"Input must have at least {self.sequence_length} words!")
            return
        
        input_sequence = words[-self.sequence_length:]
        print(f"Input sequence: {' '.join(input_sequence)}")
        
        # Encode to IDs
        encoded_input = [self.word_to_id.get(word, 0) for word in input_sequence]
        
        # Predict
        predictions = self.predictor.predict_next_words(encoded_input, top_k)
        
        print(f"\nTop {top_k} predicted next words:")
        for i, (word_id, prob) in enumerate(predictions, 1):
            word = self.id_to_word.get(word_id, '<unknown>')
            print(f"  {i}. {word} (probability: {prob:.4f})")
    
    def interactive_testing(self):
        """Interactive testing mode"""
        print(f"\n=== INTERACTIVE TESTING ===")
        print(f"Enter text with at least {self.sequence_length} words to get next word predictions.")
        print("Type 'quit' to exit.")
        
        while True:
            user_input = input(f"\nEnter text: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if not user_input:
                continue
            
            self.test_prediction(user_input)
    
    def save_system(self, model_path='model.keras', vocab_path='vocabulary.pkl'):
        """Save the trained model and vocabulary"""
        if self.predictor is None:
            print("No model to save!")
            return
        
        # Save model
        self.predictor.save_model(model_path)
        
        # Save vocabulary and mappings
        vocab_data = {
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'sequence_length': self.sequence_length
        }
        
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        print(f"Model saved to {model_path}")
        print(f"Vocabulary saved to {vocab_path}")
    
    def load_system(self, model_path='model.keras', vocab_path='vocabulary.pkl'):
        """Load a trained model and vocabulary"""
        if not os.path.exists(model_path) or not os.path.exists(vocab_path):
            print("Model or vocabulary files not found!")
            return False
        
        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.word_to_id = vocab_data['word_to_id']
        self.id_to_word = vocab_data['id_to_word']
        self.sequence_length = vocab_data['sequence_length']
        
        # Load model
        self.predictor = NextWordPredictor(len(self.word_to_id), self.sequence_length)
        self.predictor.load_model(model_path)
        
        print(f"Model and vocabulary loaded successfully!")
        return True
    
    def run_complete_pipeline(self, epochs=30, batch_size=32):
        """Run the complete training pipeline"""
        # Load and process data
        words = self.load_and_process_data()
        
        # Prepare training data
        encoded_inputs, encoded_outputs = self.prepare_training_data(words)
        
        # Train model
        results = self.train_model(encoded_inputs, encoded_outputs, epochs, batch_size)
        
        # Save system
        self.save_system()
        
        # Test with some examples
        print(f"\n=== TESTING WITH EXAMPLES ===")
        test_sentences = [
            "holmes sat up in his chair",
            "the criminal shows considerable ingenuity and",
            "watson walked into the room and",
            "the morning sun cast long shadows",
            "adventure called to those brave enough"
        ]
        
        for sentence in test_sentences:
            self.test_prediction(sentence)
        
        # Interactive testing
        self.interactive_testing()
        
        return results

def main():
    """Main execution function"""
    print("Next Word Prediction System")
    print("=" * 50)
    
    # Initialize system
    system = NextWordPredictionSystem(sequence_length=5)
    
    # Check if pre-trained model exists
    if os.path.exists('model.keras') and os.path.exists('vocabulary.pkl'):
        print("Found existing model. Loading...")
        if system.load_system():
            print("Loaded existing model successfully!")
            system.interactive_testing()
            return
    
    # Run complete training pipeline
    print("No existing model found. Starting training...")
    results = system.run_complete_pipeline(epochs=30, batch_size=32)
    
    print(f"\nTraining completed!")
    print(f"Final test accuracy: {results['evaluation']['accuracy']:.4f}")

if __name__ == "__main__":
    main()