# Next Word Prediction System

A neural network-based system for predicting the next word in a sequence, inspired by language models like GPT. This implementation uses LSTM networks to learn patterns from text data and predict probable next words.

## Project Overview

This project implements a complete next word prediction pipeline including:
- Text preprocessing and analysis
- Neural network model with LSTM architecture
- Training and evaluation framework
- Interactive prediction interface

## Features

- **Text Analysis**: Word frequency analysis, common word pairs discovery
- **Neural Network**: LSTM-based architecture with embedding layers
- **Boolean Encoding**: Converts words to boolean arrays for neural network processing
- **Data Splitting**: Automatic train/validation/test splits
- **Interactive Testing**: Real-time prediction testing
- **Model Persistence**: Save and load trained models

## Files Structure

```
├── main.py              # Main system orchestrator
├── text_processor.py    # Text preprocessing and analysis
├── neural_network.py    # Neural network model implementation
├── book.txt            # Training text data (Sherlock Holmes stories)
├── download_data.py    # Data download utility
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd New-words-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete system:
```bash
python main.py
```

This will:
1. Load and analyze the text data
2. Train the neural network model
3. Save the trained model
4. Provide interactive testing

### Individual Components

#### Text Analysis
```python
from text_processor import TextPreprocessor

preprocessor = TextPreprocessor()
with open('book.txt', 'r') as f:
    text = f.read()

words = preprocessor.tokenize(preprocessor.clean_text(text))
analysis = preprocessor.analyze_text(words)
```

#### Model Training
```python
from neural_network import ModelTrainer

trainer = ModelTrainer(vocab_size=1000, sequence_length=5)
results = trainer.train_model(input_sequences, output_words)
```

#### Prediction
```python
from main import NextWordPredictionSystem

system = NextWordPredictionSystem()
system.load_system()  # Load pre-trained model
system.test_prediction("holmes sat up in his chair")
```

## Model Architecture

The neural network uses the following architecture:

1. **Embedding Layer**: Converts word IDs to dense vectors (50 dimensions)
2. **LSTM Layer**: Processes sequences with 128 units and dropout
3. **Dense Layer**: 128 units with ReLU activation
4. **Dropout Layer**: 30% dropout for regularization
5. **Output Layer**: Softmax activation for probability distribution

## Parameters

- **Sequence Length**: 5 words (configurable)
- **Vocabulary Size**: Based on unique words in training data
- **Embedding Dimensions**: 50
- **LSTM Units**: 128
- **Training Epochs**: 30 (with early stopping)
- **Batch Size**: 32

## Training Data

The system uses "The Adventures of Sherlock Holmes" by Arthur Conan Doyle as training data. The text is preprocessed to:
- Convert to lowercase
- Remove special characters
- Filter words by minimum length
- Create word-to-ID mappings

## Results

The system analyzes the training text and provides:
- Word frequency statistics
- Common word pair analysis
- Model accuracy metrics
- Top-K prediction accuracy

## Example Usage

```python
# Initialize system
system = NextWordPredictionSystem()

# Train or load model
system.run_complete_pipeline()

# Test prediction
system.test_prediction("the criminal shows considerable ingenuity and")
# Output: Top predicted words with probabilities
```

## Interactive Mode

The system provides an interactive mode where you can:
1. Enter any text with at least 5 words
2. Get top 5 most probable next word predictions
3. See prediction confidence scores

## Extending the System

### Using Different Text Data
Replace `book.txt` with your own text file or modify `download_data.py` to fetch different sources.

### Adjusting Model Architecture
Modify `neural_network.py` to experiment with:
- Different LSTM units
- Multiple LSTM layers
- Different embedding dimensions
- Alternative architectures (GRU, Transformer)

### Changing Sequence Length
Adjust the `sequence_length` parameter to use more or fewer input words for prediction.

## Technical Details

### Text Preprocessing
- Cleans Project Gutenberg headers/footers
- Normalizes whitespace and punctuation
- Filters words by minimum length
- Creates vocabulary mappings

### Data Encoding
Two encoding methods are supported:
1. **ID Encoding**: Words mapped to integer IDs (used in implementation)
2. **Boolean Encoding**: Words as one-hot vectors (available for experimentation)

### Model Training
- Uses Adam optimizer
- Sparse categorical crossentropy loss
- Early stopping and learning rate reduction
- Train/validation/test split (70%/10%/20%)

## Performance

The system achieves reasonable performance on the Sherlock Holmes dataset:
- Vocabulary: ~600 unique words
- Training sequences: ~1200
- Typical accuracy: 30-50% (depends on text complexity)
- Top-5 accuracy: Much higher due to language patterns

## License

MIT License - see LICENSE file for details.

## Dependencies

- TensorFlow >= 2.10.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0
- Requests >= 2.25.0

## Future Improvements

- Implement attention mechanisms
- Add support for longer sequences
- Include punctuation prediction
- Add beam search for better predictions
- Support for multiple languages
- Web interface for easier interaction