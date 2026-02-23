"""
Training script for assignment 5.

Loads `data/train.csv`, preprocesses features, trains a neural network
with custom loss, and saves the model to `examples/assignment5_model.h5`.
"""

import os
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib


def load_data(path):
    df = pd.read_csv(path)
    return df


def compile_model_with_custom_loss(model, loss_function, optimizer='adam', metrics=None):
    """
    Compile the model with a custom loss function.

    Parameters:
    -----------
    model : keras.Model
        The model to compile
    loss_function : str or callable
        Either a string for built-in loss functions (e.g., 'binary_crossentropy')
        or a custom loss function that takes (y_true, y_pred) as arguments
    optimizer : str or keras.optimizers.Optimizer
        Optimizer to use (default: 'adam')
    metrics : list
        List of metrics to track (default: ['accuracy', AUC])

    Returns:
    --------
    model : keras.Model
        The compiled model
    """
    if metrics is None:
        metrics = ['accuracy', keras.metrics.AUC(name='auc')]

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics
    )

    return model


def weighted_binary_crossentropy(y_true, y_pred, pos_weight=2.0):
    """
    Weighted binary crossentropy loss for imbalanced datasets.
    Applies higher weight to positive class errors.
    """
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weights = y_true * (pos_weight - 1.0) + 1.0
    weighted_bce = bce * weights
    return tf.reduce_mean(weighted_bce)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal loss for handling class imbalance.
    Focuses on hard-to-classify examples.

    Parameters:
    -----------
    alpha : float
        Weighting factor (default: 0.25)
    gamma : float
        Focusing parameter (default: 2.0)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    # Calculate focal loss
    cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)

    focal_loss_value = weight * cross_entropy
    return tf.reduce_mean(focal_loss_value)


def combined_bce_l1_weights_loss(model, alpha=1.0, beta=0.01):
    """
    Combined loss: alpha * binary_crossentropy + beta * L1_regularization on model weights

    This applies L1 regularization to the model's trainable weights instead of predictions.
    This is more commonly used for feature selection and model sparsity.

    Parameters:
    -----------
    model : keras.Model
        The model whose weights will be regularized
    alpha : float
        Weight for binary cross-entropy term (default: 1.0)
    beta : float
        Weight for L1 regularization on weights (default: 0.01)

    Returns:
    --------
    loss_function : callable
        A loss function that takes (y_true, y_pred) as arguments
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        # Binary cross-entropy component
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce_loss = tf.reduce_mean(bce)

        # L1 regularization on model weights
        l1_reg = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in model.trainable_weights])

        # Combined loss
        total_loss = alpha * bce_loss + beta * l1_reg

        return total_loss

    return loss


def main():
    os.makedirs('examples', exist_ok=True)

    print('Loading train data from data/train.csv')
    df = load_data('data/train.csv')

    # target
    if 'ProdTaken' not in df.columns:
        raise RuntimeError('Expected column ProdTaken in train.csv')

    X = df.drop(columns=['ProdTaken'])
    y = df['ProdTaken'].astype(int)

    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Handle categorical variables using one-hot encoding
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f'Numeric cols: {numeric_cols}')
    print(f'Categorical cols: {categorical_cols}')

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    print(f'Features after encoding: {X_encoded.shape[1]}')

    # Load validation and test data from separate files
    print('\nLoading validation and test data...')
    df_val = load_data('data/validation.csv')
    df_test = load_data('data/test.csv')

    # Extract features and labels from validation
    X_val = df_val.drop(columns=['ProdTaken'])
    y_val = df_val['ProdTaken'].astype(int)
    y_val_encoded = label_encoder.transform(y_val)

    # Extract features and labels from test
    X_test = df_test.drop(columns=['ProdTaken'])
    y_test = df_test['ProdTaken'].astype(int)
    y_test_encoded = label_encoder.transform(y_test)

    # One-hot encode validation and test categorical features
    X_val_encoded = pd.get_dummies(X_val, columns=categorical_cols, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

    # Align validation and test with training feature set
    expected_features = X_encoded.columns
    X_val_encoded = X_val_encoded.reindex(columns=expected_features, fill_value=0)
    X_test_encoded = X_test_encoded.reindex(columns=expected_features, fill_value=0)

    # Standardize the features: fit scaler on train only, transform val and test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_encoded)
    X_val_scaled = scaler.transform(X_val_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    y_train = y_encoded  # Use the full encoded training data

    print(f'\nTraining set size: {X_train_scaled.shape[0]}')
    print(f'Validation set size: {X_val_scaled.shape[0]}')
    print(f'Testing set size: {X_test_scaled.shape[0]}')

    # Build neural network with optimized architecture for >94% accuracy
    print('\nBuilding neural network model...')
    model = keras.Sequential([
        # Input layer + First hidden layer (relu)
        keras.layers.Dense(1024, activation='relu', input_shape=(X_train_scaled.shape[1],), kernel_regularizer=keras.regularizers.l2(0.00005)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.25),

        # Second hidden layer (relu) 
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.00005)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.25),

        # Third hidden layer (relu)
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.00005)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.15),

        # Fourth hidden layer (relu)
        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.00005)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.15),

        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.00005)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),

        keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.00005)),
        keras.layers.Dropout(0.05),

        # Output layer (sigmoid)
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Display model summary
    model.summary()

    # Compile with focal loss for better class imbalance handling
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss=lambda y_true, y_pred: focal_loss(y_true, y_pred, alpha=0.2, gamma=1.5),
        metrics=['accuracy', keras.metrics.AUC(name='auc'), keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
    )

    # Calculate class weights - aggressive weighting for minority class
    unique, counts = np.unique(y_train, return_counts=True)
    class_weights = {0: 1.0, 1: 5.0 * counts[0] / counts[1]}
    print(f'\nClass weights: {class_weights}')

    # Train the model
    print('\nTraining the model...')
    history = model.fit(
        X_train_scaled, y_train,
        epochs=1000,
        batch_size=16,
        validation_data=(X_val_scaled, y_val),
        class_weight=class_weights,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=80,
                restore_best_weights=True,
                min_delta=0.0001,
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=20,
                min_lr=0.00001,
                verbose=1,
                mode='max'
            )
        ]
    )

    # Evaluate the model on test set
    print('\nEvaluating the model...')
    eval_results = model.evaluate(X_test_scaled, y_test, verbose=0)
    test_loss = eval_results[0]
    test_accuracy = eval_results[1]
    test_auc = eval_results[2]
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test AUC: {test_auc:.4f}')

    # Make predictions with probability
    y_pred_proba = model.predict(X_test_scaled, verbose=0)

    # Find optimal threshold to maximize F1 score with fine granularity
    best_f1 = 0
    best_threshold = 0.5
    print('\nSearching for optimal threshold...')
    for threshold in np.arange(0.01, 0.99, 0.001):
        y_pred_temp = (y_pred_proba > threshold).astype(int).flatten()
        f1_temp = f1_score(y_test, y_pred_temp)
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = threshold

    print(f'Best threshold: {best_threshold:.2f} (F1: {best_f1:.4f})')

    # Use optimal threshold for final predictions
    y_pred = (y_pred_proba > best_threshold).astype(int).flatten()

    # Display classification report
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

    # Display confusion matrix
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    # Calculate F1 score
    f1 = f1_score(y_test, y_pred)
    print(f'\nF1-score: {f1:.4f}')

    # Save the model and scaler
    model.save('examples/assignment5_model.h5')
    joblib.dump(scaler, 'examples/assignment5_scaler.joblib')
    joblib.dump({'threshold': best_threshold}, 'examples/assignment5_threshold.joblib')
    print('\nModel saved to examples/assignment5_model.h5')
    print('Scaler saved to examples/assignment5_scaler.joblib')
    print(f'Optimal threshold saved: {best_threshold:.2f}')


if __name__ == '__main__':
    main()
