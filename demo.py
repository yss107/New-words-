#!/usr/bin/env python3
"""
Demonstration script showing the complete Next Word Prediction system
"""
import os
import sys
import time

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def demonstrate_system():
    """Run a complete demonstration"""
    print_header("NEXT WORD PREDICTION SYSTEM DEMONSTRATION")
    
    print("\nThis system implements a neural network-based next word prediction")
    print("model similar to GPT, but simpler. It learns from text data and")
    print("predicts the most likely next word given a sequence of input words.")
    
    # Import here to avoid early imports
    from text_processor import TextPreprocessor, DatasetGenerator
    from neural_network import NextWordPredictor, ModelTrainer
    
    print_header("STEP 1: TEXT ANALYSIS")
    
    # Load and analyze text
    with open('book.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    preprocessor = TextPreprocessor()
    cleaned_text = preprocessor.clean_text(text)
    words = preprocessor.tokenize(cleaned_text)
    preprocessor.build_vocabulary(words)
    analysis = preprocessor.analyze_text(words)
    
    print(f"📖 Source: The Adventures of Sherlock Holmes")
    print(f"📊 Total words: {analysis['total_words']:,}")
    print(f"📝 Unique words: {analysis['unique_words']:,}")
    print(f"📏 Average word length: {analysis['avg_word_length']:.1f} characters")
    
    print(f"\n🔤 Most frequent words:")
    for i, (word, count) in enumerate(analysis['most_common_words'][:5], 1):
        print(f"   {i}. '{word}' appears {count} times")
    
    print(f"\n🔗 Most common word pairs:")
    for i, ((w1, w2), count) in enumerate(analysis['most_common_pairs'][:5], 1):
        print(f"   {i}. '{w1} {w2}' appears {count} times")
    
    print_header("STEP 2: TRAINING DATA PREPARATION")
    
    # Prepare training data
    generator = DatasetGenerator(sequence_length=5)
    input_sequences, output_words = generator.create_sequences(words)
    encoded_inputs = generator.encode_sequences(input_sequences, preprocessor.word_to_id)
    encoded_outputs = generator.encode_words(output_words, preprocessor.word_to_id)
    
    print(f"🔄 Sequence length: 5 words")
    print(f"📋 Training sequences created: {len(encoded_inputs):,}")
    print(f"🎯 Vocabulary size: {len(preprocessor.word_to_id):,}")
    
    print(f"\n📝 Example training sequence:")
    print(f"   Input:  {' '.join(input_sequences[0])}")
    print(f"   Output: {output_words[0]}")
    
    print_header("STEP 3: NEURAL NETWORK TRAINING")
    
    print("🧠 Building LSTM neural network...")
    print("   • Embedding layer (50 dimensions)")
    print("   • LSTM layer (128 units)")
    print("   • Dense layer (128 units)")
    print("   • Dropout (30%)")
    print("   • Output layer (softmax)")
    
    # Create and train model (quick version for demo)
    trainer = ModelTrainer(len(preprocessor.word_to_id), sequence_length=5)
    
    print(f"\n🚀 Training model (quick demo with 3 epochs)...")
    results = trainer.train_model(
        encoded_inputs, encoded_outputs, 
        epochs=3, batch_size=16
    )
    
    accuracy = results['evaluation']['accuracy']
    print(f"\n✅ Training completed!")
    print(f"   Final accuracy: {accuracy:.1%}")
    print(f"   (Note: Low accuracy is normal with only 3 epochs)")
    
    print_header("STEP 4: PREDICTION DEMONSTRATION")
    
    predictor = results['predictor']
    
    # Test sentences from the training data domain
    test_sentences = [
        "holmes sat up in his chair",
        "the criminal shows considerable ingenuity and", 
        "watson walked into the room and",
        "the morning sun cast long shadows",
        "in the smoking room of the"
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n🔮 Test {i}: '{sentence}'")
        
        # Tokenize and prepare
        test_words = preprocessor.tokenize(sentence)
        if len(test_words) >= 5:
            input_seq = test_words[-5:]
            encoded_seq = [preprocessor.word_to_id.get(w, 0) for w in input_seq]
            
            # Predict
            predictions = predictor.predict_next_words(encoded_seq, top_k=3)
            
            print(f"   📥 Input: {' '.join(input_seq)}")
            print(f"   📤 Predicted next words:")
            
            for j, (word_id, prob) in enumerate(predictions, 1):
                word = preprocessor.id_to_word.get(word_id, '<unknown>')
                print(f"      {j}. '{word}' ({prob:.1%})")
    
    print_header("SYSTEM CAPABILITIES")
    
    print("✨ What this system demonstrates:")
    print("   • Text preprocessing and analysis")
    print("   • Vocabulary building and word encoding")
    print("   • LSTM neural network architecture")
    print("   • Sequence-to-word prediction")
    print("   • Probability distributions over vocabulary")
    print("   • Model training and evaluation pipeline")
    
    print("\n🚀 How to use:")
    print("   • Run 'python main.py' for full training (30 epochs)")
    print("   • Run 'python test_system.py' for quick validation")
    print("   • Modify hyperparameters in the code")
    print("   • Replace 'book.txt' with your own text data")
    
    print("\n📈 For better results:")
    print("   • Train for more epochs (30-50)")
    print("   • Use larger text datasets")
    print("   • Experiment with model architecture")
    print("   • Adjust sequence length")
    
    print_header("DEMONSTRATION COMPLETE")
    print("🎉 Next Word Prediction system successfully demonstrated!")
    print("   Check the README.md for detailed usage instructions.")
    
    return True

if __name__ == "__main__":
    try:
        demonstrate_system()
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)