import numpy as np
import pandas as pd
from algorithms import *

def read_data(trainfile='MNIST_train.csv', validationfile='MNIST_validation.csv'):
    dftrain = pd.read_csv(trainfile)
    dfval = pd.read_csv(validationfile)
    featurecols = list(dftrain.columns)
    featurecols.remove('label')
    featurecols.remove('even')
    Xtrain = dftrain[featurecols]
    ytrain = dftrain['label']
    Xval = dfval[featurecols]
    yval = dfval['label']
    return Xtrain, ytrain, Xval, yval

def preprocess_data(Xtrain, Xval, ytrain):
    scaler = NumPyScaler()
    Xtrain_scaled = scaler.fit_transform(Xtrain)
    Xval_scaled = scaler.transform(Xval)

    pca_175 = NumPyPCA(n_components=175)
    Xtrain_pca_175 = pca_175.fit_transform(Xtrain_scaled)
    Xval_pca_175 = pca_175.transform(Xval_scaled)

    pca_50 = NumPyPCA(n_components=50)
    Xtrain_pca_50 = pca_50.fit_transform(Xtrain_scaled)
    Xval_pca_50 = pca_50.transform(Xval_scaled)

    lda_9 = NumPyLDA(n_components=9)
    Xtrain_lda_9 = lda_9.fit_transform(Xtrain_scaled, ytrain)
    Xval_lda_9 = lda_9.transform(Xval_scaled)

    return {
        'Xtrain_pca_175': Xtrain_pca_175, 'Xval_pca_175': Xval_pca_175,
        'Xtrain_pca_50': Xtrain_pca_50, 'Xval_pca_50': Xval_pca_50,
        'Xtrain_lda_9': Xtrain_lda_9, 'Xval_lda_9': Xval_lda_9
    }

def build_and_run_ensemble(
    X_train_dict, X_val_dict, y_train, y_val,
    softmax_cfg=None, knn_pca_cfg=None, knn_lda_cfg=None,
    meta_cfg=None, folds=3
):

    softmax_cfg = softmax_cfg or {'n_epochs': 100, 'learning_rate': 0.05, 'alpha': 0.01, 'batch_size': 128}
    knn_pca_cfg = knn_pca_cfg or {'k': 5, 'weights': 'distance'}
    knn_lda_cfg = knn_lda_cfg or {'k': 11, 'weights': 'distance'}
    meta_cfg = meta_cfg or {'n_epochs': 200, 'learning_rate': 0.05, 'alpha': 0.01, 'batch_size': 64}

    softmax_model = SoftmaxClassifier(**softmax_cfg)
    knn_model_pca = NumPyKNNClassifier(**knn_pca_cfg)
    knn_model_lda = NumPyKNNClassifier(**knn_lda_cfg)

    base_learners = [
        ('softmax_pca_175', softmax_model, 'pca_175'),
        ('knn_pca_50', knn_model_pca, 'pca_50'),
        ('knn_lda_9', knn_model_lda, 'lda_9')
    ]

    meta_model = SoftmaxClassifier(**meta_cfg)

    ensemble = KFoldStackingEnsemble(base_models=base_learners, meta_model=meta_model, k_folds=folds)
    ensemble.fit(X_train_dict, y_train)

    val_pred = ensemble.predict(X_val_dict)
    train_pred = ensemble.predict(X_train_dict)

    metrics = {
        'train_acc': numpy_accuracy_score(y_train, train_pred),
        'train_f1': numpy_f1_score(y_train, train_pred, average='macro'),
        'val_acc': numpy_accuracy_score(y_val, val_pred),
        'val_f_1': numpy_f1_score(y_val, val_pred, average='macro')
    }

    return {
        'ensemble': ensemble,
        'base_learners': base_learners,
        'meta_model': meta_model,
        'predictions': {'train': train_pred, 'val': val_pred},
        'metrics': metrics
    }

if __name__ == "__main__":
    Xtrain, ytrain, Xval, yval = read_data('MNIST_train.csv', 'MNIST_validation.csv')
    pre = preprocess_data(Xtrain, Xval, ytrain)

    X_train_dict = {
        'pca_175': pre['Xtrain_pca_175'],
        'pca_50':  pre['Xtrain_pca_50'],
        'lda_9':   pre['Xtrain_lda_9']
    }
    X_val_dict = {
        'pca_175': pre['Xval_pca_175'],
        'pca_50':  pre['Xval_pca_50'],
        'lda_9':   pre['Xval_lda_9']
    }

    results = build_and_run_ensemble(X_train_dict, X_val_dict, ytrain, yval, folds=3)
    print("\n--- Model Metrics ---")
    print(f"Train Accuracy: {results['metrics']['train_acc']:.4f}")
    print(f"Train F1 Macro: {results['metrics']['train_f1']:.4f}")
    print(f"Val Accuracy:   {results['metrics']['val_acc']:.4f}")
    print(f"Val F1 Macro:   {results['metrics']['val_f_1']:.4f}")
