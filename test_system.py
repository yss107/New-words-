#!/usr/bin/env python3
"""
Quick test script for the Next Word Prediction System
"""
import os
import sys
from main import NextWordPredictionSystem

def quick_test():
    """Run a quick test with fewer epochs"""
    print("Quick Test of Next Word Prediction System")
    print("=" * 50)
    
    # Initialize system
    system = NextWordPredictionSystem(sequence_length=5)
    
    # Load and process data
    words = system.load_and_process_data()
    
    # Prepare training data 
    encoded_inputs, encoded_outputs = system.prepare_training_data(words)
    
    # Train with fewer epochs for quick test
    print(f"\nTraining with reduced epochs for quick test...")
    results = system.train_model(encoded_inputs, encoded_outputs, epochs=5, batch_size=16)
    
    print(f"\nTraining completed!")
    print(f"Final test accuracy: {results['evaluation']['accuracy']:.4f}")
    
    # Test a few predictions
    print(f"\n=== TESTING PREDICTIONS ===")
    test_sentences = [
        "holmes sat up in his chair",
        "the criminal shows considerable ingenuity and",
        "watson walked into the room and"
    ]
    
    for sentence in test_sentences:
        system.test_prediction(sentence, top_k=3)
    
    return True

if __name__ == "__main__":
    try:
        success = quick_test()
        if success:
            print("\n✅ Quick test completed successfully!")
        else:
            print("\n❌ Quick test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        sys.exit(1)