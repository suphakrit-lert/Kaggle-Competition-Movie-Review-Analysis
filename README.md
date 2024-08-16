# Kaggle Competition: Movie Review Preference Analysis


## Description
Sentiment analysis is a fascinating field in natural language processing (NLP) that deals with understanding the emotions conveyed in written text. As a natural extension to this established problem, in this competition, you will build a model that automatically detects the more positive review in any pairs of movie reviews.

Specifically, for every pair of reviews (r0, r1), you are given the feature vectors (E(r0), E(r1)) encoded by a language model. The goal is to build a model that can predict y, the binary label representing the "preference" between these two reviews. You predict 1 if r0 is more positive than r1 and 0 otherwise. We use the Accuracy of the test data as the evaluation metric.

## Dataset
To create a movie review preference dataset, we leveraged an existing movie review dataset that contains sets of positive and negative reviews for a large amount of films. We randomly paired positive and negative reviews together as ( $r \theta, r 1$ ). To synthesize preference annotations, we assume that positive reviews are preferred over negative ones and assigned the preference label 1 to samples with a preceding positive review; 0 to ones that begin with negative reviews. We then extracted features for each review using an advanced language model and provided these as the sole inputs for your models.

### Dataset Splits
We have two partitions: a training set with 18,750 samples and a test set with 6,250 samples. We provide both features ( $E(r \theta), E(r 1))$ and labels $y$ for the training set, while the test set only has the features. Note that the split occurs at the review level, so it is possible that two reviews of the same movie end up encoded in feature vectors of different splits.

### Files
- `train.npz`: The training data with the following columns:
  - `uid`: Unique ID across the entire dataset.
  - `emb1`: Language model-encoded feature vector of the review r0. Dimension $=384$.
  - `emb2 `: Language model-encoded feature vector of the review $r1$. Dimension $=384$.
  - `preference`: Binary label signifying whether the first review ( $r0$ ) is preferred (preference=1) or the second (preference=0).
- `test.npz`: The test data.
  - `uid`: Unique ID across the entire dataset.
  - `emb1, emb2`: Same as above.

## Approach
- **Data Preparation**: Gather and preprocess data, divide into training and validation sets using PyTorch's batch dataloader function.
- **Network Architecture**: Define the network architecture with specified layers and neurons:
  - Input layer: Represents features.
  - Output layer: Corresponds to predictions or classifications.
  - Hidden layers: Contain neurons with weighted connections, activation functions introduce non-linearity.
  - Initialize weights and biases randomly.
- **Training Setup**: 
  - Choose loss function (cross-entropy for classification) and optimizer (Adam).
- **Training Process**: 
  - Train the network with labeled data.
  - Adjust weights and biases to minimize the difference between predictions and actual values.
  - Continue training until the model converges and achieves minimal loss.
- **Hyperparameter Tuning**: Use grid search to find optimal parameters, including:
  - Hidden dimensions
  - Activation functions (ReLU, sigmoid, tanh)
  - Number of layers
  - Batch normalization
  - Weight initialization
  - Dropout rates
  - Learning rates (alphas)
  - Number of epochs

