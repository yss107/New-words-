#!/usr/bin/env python3
"""
Neural Network Model for Next Word Prediction
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
import pickle
import os

class NextWordPredictor:
    """Neural network model for next word prediction"""
    
    def __init__(self, vocab_size: int, sequence_length: int = 5, embedding_dim: int = 50):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.model = None
        
    def build_model(self) -> keras.Model:
        """Build the neural network architecture"""
        model = keras.Sequential([
            # Embedding layer to convert word IDs to dense vectors
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.sequence_length,
                name='embedding'
            ),
            
            # LSTM layer for sequence processing
            layers.LSTM(
                units=128,
                dropout=0.2,
                recurrent_dropout=0.2,
                name='lstm_1'
            ),
            
            # Dense layer with ReLU activation
            layers.Dense(
                units=128,
                activation='relu',
                name='dense_1'
            ),
            
            # Dropout for regularization
            layers.Dropout(0.3, name='dropout'),
            
            # Output layer with softmax for probability distribution
            layers.Dense(
                units=self.vocab_size,
                activation='softmax',
                name='output'
            )
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, input_sequences: List[List[int]], output_words: List[int], 
                    test_size: float = 0.2, validation_size: float = 0.1) -> Tuple:
        """Prepare training, validation, and test datasets"""
        X = np.array(input_sequences)
        y = np.array(output_words)
        
        # Split into train and temp (test + validation)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + validation_size), random_state=42
        )
        
        # Split temp into test and validation
        val_ratio = validation_size / (test_size + validation_size)
        X_test, X_val, y_test, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def train(self, train_data: Tuple, val_data: Tuple, 
              epochs: int = 50, batch_size: int = 32, verbose: int = 1) -> keras.callbacks.History:
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def evaluate(self, test_data: Tuple) -> Dict:
        """Evaluate the model on test data"""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        X_test, y_test = test_data
        
        # Get predictions
        predictions = self.model.predict(X_test, verbose=0)
        
        # Calculate metrics
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Top predictions analysis
        top_predictions = np.argsort(predictions, axis=1)[:, -5:]  # Top 5 predictions
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'top_predictions': top_predictions
        }
    
    def predict_next_words(self, input_sequence: List[int], top_k: int = 5) -> List[Tuple[int, float]]:
        """Predict the next word given an input sequence"""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare input
        X = np.array([input_sequence])
        
        # Get predictions
        predictions = self.model.predict(X, verbose=0)[0]
        
        # Get top k predictions with probabilities
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_probs = predictions[top_indices]
        
        return list(zip(top_indices, top_probs))
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)

class ModelTrainer:
    """Helper class to manage the complete training pipeline"""
    
    def __init__(self, vocab_size: int, sequence_length: int = 5):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.predictor = NextWordPredictor(vocab_size, sequence_length)
        
    def train_model(self, input_sequences: List[List[int]], output_words: List[int],
                   epochs: int = 50, batch_size: int = 32) -> Dict:
        """Complete training pipeline"""
        print("Preparing data...")
        train_data, val_data, test_data = self.predictor.prepare_data(
            input_sequences, output_words
        )
        
        print(f"Training data: {len(train_data[0])} samples")
        print(f"Validation data: {len(val_data[0])} samples")
        print(f"Test data: {len(test_data[0])} samples")
        
        print("\nBuilding model...")
        self.predictor.build_model()
        print(self.predictor.model.summary())
        
        print("\nTraining model...")
        history = self.predictor.train(
            train_data, val_data, epochs=epochs, batch_size=batch_size
        )
        
        print("\nEvaluating model...")
        evaluation = self.predictor.evaluate(test_data)
        
        print(f"Test Loss: {evaluation['loss']:.4f}")
        print(f"Test Accuracy: {evaluation['accuracy']:.4f}")
        
        return {
            'history': history,
            'evaluation': evaluation,
            'predictor': self.predictor
        }

def create_boolean_encoding(input_sequences: List[List[int]], output_words: List[int], 
                          vocab_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create boolean array encoding for neural network"""
    # Input encoding: each sequence becomes a 2D boolean array
    X_bool = np.zeros((len(input_sequences), len(input_sequences[0]), vocab_size), dtype=bool)
    for i, sequence in enumerate(input_sequences):
        for j, word_id in enumerate(sequence):
            if 0 <= word_id < vocab_size:
                X_bool[i, j, word_id] = True
    
    # Output encoding: each word becomes a 1D boolean array
    y_bool = np.zeros((len(output_words), vocab_size), dtype=bool)
    for i, word_id in enumerate(output_words):
        if 0 <= word_id < vocab_size:
            y_bool[i, word_id] = True
    
    return X_bool, y_bool

def main():
    """Demo the neural network training"""
    print("Neural Network Model Demo")
    print("This module provides the NextWordPredictor class for training and prediction.")
    
    # Example usage
    vocab_size = 100
    sequence_length = 5
    
    # Create dummy data for demo
    np.random.seed(42)
    n_samples = 1000
    input_sequences = [
        [np.random.randint(0, vocab_size) for _ in range(sequence_length)]
        for _ in range(n_samples)
    ]
    output_words = [np.random.randint(0, vocab_size) for _ in range(n_samples)]
    
    print(f"\nDemo with {n_samples} samples, vocab_size={vocab_size}")
    
    # Create boolean encoding
    X_bool, y_bool = create_boolean_encoding(input_sequences, output_words, vocab_size)
    print(f"Boolean encoding shapes: X={X_bool.shape}, y={y_bool.shape}")
    
    # Create trainer
    trainer = ModelTrainer(vocab_size, sequence_length)
    print("\nModel architecture created successfully!")

if __name__ == "__main__":
    main()