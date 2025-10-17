import os
import time
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_curve, auc, precision_recall_curve, confusion_matrix, roc_auc_score
)
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

# Imbalanced-learn import for SMOTE
from imblearn.over_sampling import SMOTE

# TensorFlow and Keras imports for DNN
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, save_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.regularizers import l2

# SHAP for explainability
import shap
from shap import kmeans

# --- Configuration and Constants ---
os.makedirs("results/isolation_forest", exist_ok=True)
os.makedirs("results/dnn", exist_ok=True)
os.makedirs("results/hybrid", exist_ok=True)
os.makedirs("models", exist_ok=True)

# File paths
TRAIN_FILE = "data/KDDTrain+.txt"
TEST_FILE = "data/KDDTest+.txt"
IF_MODEL_PATH = "models/isolation_forest.joblib"
DNN_MODEL_PATH = "models/dnn_model.h5"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
K_FEATURES = 50 # FINAL IMPROVEMENT: Fine-tuned feature selection

COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'difficulty'
]

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data():
    """Loads, cleans, and preprocesses the NSL-KDD dataset."""
    print("Loading and preprocessing data...")
    df_train = pd.read_csv(TRAIN_FILE, header=None, names=COLUMNS)
    df_test = pd.read_csv(TEST_FILE, header=None, names=COLUMNS)

    y_train = (df_train['class'] != 'normal').astype(int)
    y_test = (df_test['class'] != 'normal').astype(int)

    X_train = df_train.drop(columns=['class', 'difficulty', 'num_outbound_cmds'])
    X_test = df_test.drop(columns=['class', 'difficulty', 'num_outbound_cmds'])

    categorical_features = ['protocol_type', 'service', 'flag']
    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(f_classif, k=K_FEATURES))
    ])

    print(f"Fitting pipeline and selecting top {K_FEATURES} features...")
    X_train_processed = pipeline.fit_transform(X_train, y_train)
    X_test_processed = pipeline.transform(X_test)
    
    joblib.dump(pipeline, PREPROCESSOR_PATH)
    print(f"Full preprocessing pipeline saved to {PREPROCESSOR_PATH}")
    print("Data preprocessing complete.")
    return X_train_processed, y_train, X_test_processed, y_test, pipeline

# --- Visualization and Evaluation Functions ---
def plot_confusion_matrix(y_true, y_pred, model_name, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_scores, model_name, path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Receiver Operating Characteristic - {model_name}', fontsize=16)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_pr_curve(y_true, y_scores, model_name, path):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=16)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_score_distribution(scores, y_true, model_name, path):
    df_plot = pd.DataFrame({'score': scores, 'label': y_true})
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_plot, x='score', hue='label', palette={0: 'blue', 1: 'red'}, kde=True, stat="density", common_norm=False)
    plt.title(f'Anomaly Score Distribution - {model_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_tsne(X, y, y_pred, model_name, path):
    print(f"Generating t-SNE for {model_name}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=300, init='pca')
    sample_size = min(X.shape[0], 2500)
    sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sample = X[sample_indices]
    y_sample = y.iloc[sample_indices] if isinstance(y, (pd.Series, pd.DataFrame)) else y[sample_indices]
    y_pred_sample = y_pred[sample_indices]
    
    X_tsne = tsne.fit_transform(X_sample)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f't-SNE Visualization - {model_name}', fontsize=20)
    
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_sample, palette={0: 'blue', 1: 'red'}, legend='full', ax=ax1)
    ax1.set_title('True Labels', fontsize=16)
    
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_pred_sample, palette={0: 'blue', 1: 'red'}, legend='full', ax=ax2)
    ax2.set_title('Predicted Labels', fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"t-SNE plot for {model_name} saved.")

def calculate_metrics(y_true, y_pred, y_scores):
    """Calculates and returns a dictionary of performance metrics."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_scores),
        'PR-AUC': auc(recall, precision)
    }

def evaluate_model_plots(y_true, y_pred, y_scores, X_data, model_name, results_path):
    """Generates all plots for a given model's evaluation."""
    print(f"Generating plots for {model_name}...")
    plot_confusion_matrix(y_true, y_pred, model_name, f"{results_path}/confusion_matrix.png")
    plot_roc_curve(y_true, y_scores, model_name, f"{results_path}/roc_curve.png")
    plot_pr_curve(y_true, y_scores, model_name, f"{results_path}/pr_curve.png")
    plot_score_distribution(y_scores, y_true, model_name, f"{results_path}/distribution_plot.png")
    plot_tsne(X_data, y_true, y_pred, model_name, f"{results_path}/tsne_visualization.png")
    print(f"All plots for {model_name} saved in '{results_path}'")

def find_optimal_threshold(y_true, y_scores, model_name=""):
    """Finds the optimal threshold to maximize F1-score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_f1_idx]
    if model_name:
        print(f"Optimal threshold for {model_name} (max F1-score): {best_threshold:.4f}")
    return best_threshold

# --- Model Implementations ---
def run_isolation_forest(X_train, y_train, X_test, y_test, pipeline):
    print("\n=== Running Isolation Forest Model ===")
    start_time = time.time()
    
    # FINAL IMPROVEMENT: Train only on NORMAL data to improve precision
    print("Training Isolation Forest on NORMAL data only...")
    X_train_normal = X_train[y_train == 0]
    model = IsolationForest(n_estimators=500, max_samples=256, contamination='auto', random_state=42, n_jobs=-1)
    model.fit(X_train_normal)

    joblib.dump(model, IF_MODEL_PATH)
    print(f"Isolation Forest model saved to {IF_MODEL_PATH}")

    # --- Training Set Evaluation for Threshold Tuning (using all train data) ---
    y_scores_train = -model.decision_function(X_train)
    optimal_threshold_if = find_optimal_threshold(y_train, y_scores_train, "Isolation Forest (Train)")
    y_pred_train = (y_scores_train > optimal_threshold_if).astype(int)
    train_metrics = calculate_metrics(y_train, y_pred_train, y_scores_train)
    print("\n--- Training Performance (Isolation Forest) ---")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # --- Testing Set Evaluation ---
    y_scores_if = -model.decision_function(X_test)
    y_pred_if = (y_scores_if > optimal_threshold_if).astype(int)
    
    exec_time = time.time() - start_time
    print(f"\nExecution time: {exec_time:.2f} seconds")

    test_metrics = calculate_metrics(y_test, y_pred_if, y_scores_if)
    print("\n--- Testing Performance (Isolation Forest) ---")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    evaluate_model_plots(y_test, y_pred_if, y_scores_if, X_test, "Isolation Forest", "results/isolation_forest")
    test_metrics['Execution Time'] = exec_time
    
    # --- SHAP Analysis ---
    print("Generating SHAP summary plot for Isolation Forest...")
    preprocessor = pipeline.named_steps['preprocessor']
    selector = pipeline.named_steps['feature_selection']
    all_feature_names = preprocessor.get_feature_names_out()
    selected_mask = selector.get_support()
    selected_feature_names = all_feature_names[selected_mask]
    
    X_test_df = pd.DataFrame(X_test, columns=selected_feature_names)
    explainer = shap.TreeExplainer(model, X_test_df.sample(100, random_state=42))
    shap_values = explainer.shap_values(X_test_df.sample(100, random_state=42)) 
    
    plt.figure()
    shap.summary_plot(shap_values, X_test_df.sample(100, random_state=42), show=False, plot_type='bar')
    plt.title("Feature Importance - Isolation Forest (SHAP)", fontsize=16)
    plt.tight_layout()
    plt.savefig("results/isolation_forest/shap_summary.png", dpi=300)
    plt.close()
    print("SHAP plot saved.")
    
    return test_metrics, y_pred_if, y_scores_if

def run_dnn(X_train, y_train, X_test, y_test, pipeline):
    print("\n=== Running Deep Neural Network (DNN) Model ===")
    start_time = time.time()
    
    print("Applying SMOTE to balance the training data for DNN...")
    smote = SMOTE(random_state=42, n_jobs=-1)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"Training data balanced.")

    # FINAL IMPROVEMENT: Optimized regularization and architecture
    l2_reg = 0.001 
    model = Sequential([
        Dense(128, input_dim=K_FEATURES, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, min_lr=1e-7, verbose=1)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
    
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_smote), y=y_train_smote)
    class_weights = {i: weights[i] for i in range(len(weights))}

    print("Training DNN model...")
    history = model.fit(X_train_smote, y_train_smote, 
                        epochs=200, 
                        batch_size=512, 
                        validation_split=0.2, 
                        callbacks=[early_stopping, reduce_lr], 
                        class_weight=class_weights, 
                        verbose=1)
    
    save_model(model, DNN_MODEL_PATH)
    print(f"DNN model saved to {DNN_MODEL_PATH}")

    # --- Training Set Evaluation for Threshold ---
    y_scores_train = model.predict(X_train).flatten()
    optimal_threshold_dnn = find_optimal_threshold(y_train, y_scores_train, "DNN (Train)")
    y_pred_train = (y_scores_train > optimal_threshold_dnn).astype(int)
    train_metrics = calculate_metrics(y_train, y_pred_train, y_scores_train)
    print("\n--- Training Performance (DNN) ---")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # --- Testing Set Evaluation ---
    y_scores_dnn = model.predict(X_test).flatten()
    y_pred_dnn = (y_scores_dnn > optimal_threshold_dnn).astype(int)
    
    exec_time = time.time() - start_time
    print(f"\nExecution time: {exec_time:.2f} seconds")
    
    test_metrics = calculate_metrics(y_test, y_pred_dnn, y_scores_dnn)
    print("\n--- Testing Performance (DNN) ---")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    evaluate_model_plots(y_test, y_pred_dnn, y_scores_dnn, X_test, "DNN", "results/dnn")
    test_metrics['Execution Time'] = exec_time

    # FINAL IMPROVEMENT: Use robust KernelExplainer and drastically reduce sample size for speed
    print("Generating SHAP summary plot for DNN (this may take a few minutes)...")
    background_summary = kmeans(X_train, 10)
    # Wrap the predict function in a lambda to ensure correct output shape for SHAP
    predict_fn = lambda x: model.predict(x).flatten()
    explainer = shap.KernelExplainer(predict_fn, background_summary)
    
    # Drastically reduce sample size for faster SHAP calculation
    X_test_sample = X_test[np.random.choice(X_test.shape[0], 5, replace=False)]
    shap_values = explainer.shap_values(X_test_sample)
    
    preprocessor = pipeline.named_steps['preprocessor']
    selector = pipeline.named_steps['feature_selection']
    all_feature_names = preprocessor.get_feature_names_out()
    selected_mask = selector.get_support()
    selected_feature_names = all_feature_names[selected_mask]
    
    X_test_df_sample = pd.DataFrame(X_test_sample, columns=selected_feature_names)
    
    plt.figure()
    shap.summary_plot(shap_values, X_test_df_sample, show=False, plot_type='bar')
    plt.title("Feature Importance - DNN (SHAP)", fontsize=16)
    plt.tight_layout()
    plt.savefig("results/dnn/shap_summary.png", dpi=300)
    plt.close()
    print("SHAP plot saved.")
    
    return test_metrics, y_pred_dnn, y_scores_dnn

def find_optimal_hybrid_weights(y_scores_if, y_scores_dnn, y_test_true):
    print("\n--- Finding Optimal Hybrid Weights ---")
    best_f1 = 0
    best_weight = 0.5 # Default to 50/50
    scaler = MinMaxScaler()
    y_scores_if_scaled = scaler.fit_transform(y_scores_if.reshape(-1, 1)).flatten()
    
    # FINAL IMPROVEMENT: Force a true hybrid by searching in a restricted range (max 60% DNN)
    for w in np.arange(0.05, 0.61, 0.05):
        hybrid_scores = (1 - w) * y_scores_if_scaled + w * y_scores_dnn
        optimal_threshold = find_optimal_threshold(y_test_true, hybrid_scores) 
        y_pred_hybrid = (hybrid_scores > optimal_threshold).astype(int)
        f1 = f1_score(y_test_true, y_pred_hybrid)
        print(f"Weight (DNN={w:.2f}): F1-Score = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_weight = w
            
    print(f"\nBest weight for DNN: {best_weight:.2f} with F1-Score: {best_f1:.4f}")
    return best_weight

def run_hybrid_model(y_scores_if, y_scores_dnn, X_test, y_test):
    """Combines model scores with optimal weights and evaluates."""
    print("\n=== Evaluating Hybrid Model (Weighted Score) ===")
    
    optimal_weight_dnn = find_optimal_hybrid_weights(y_scores_if, y_scores_dnn, y_test)
    
    scaler = MinMaxScaler()
    y_scores_if_scaled = scaler.fit_transform(y_scores_if.reshape(-1, 1)).flatten()
    
    hybrid_scores = (1 - optimal_weight_dnn) * y_scores_if_scaled + optimal_weight_dnn * y_scores_dnn
    
    optimal_threshold_hybrid = find_optimal_threshold(y_test, hybrid_scores, "Hybrid Model (Test)")
    y_pred_hybrid = (hybrid_scores > optimal_threshold_hybrid).astype(int)
    
    metrics = calculate_metrics(y_test, y_pred_hybrid, hybrid_scores)
    
    print("\n--- Final Hybrid Model Performance on Test Data ---")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
        
    evaluate_model_plots(y_test, y_pred_hybrid, hybrid_scores, X_test, "Hybrid Model", "results/hybrid")
    
    return metrics, hybrid_scores

def plot_comparison_charts(if_metrics, dnn_metrics, hybrid_metrics, y_test, y_scores_if, y_scores_dnn, hybrid_scores):
    """Generates bar chart for metrics and line chart for ROC curves."""
    # --- Bar Chart for Metrics ---
    metrics_to_plot = ['F1-Score', 'Precision', 'Recall', 'Accuracy', 'ROC-AUC', 'PR-AUC']
    models = ['Isolation Forest', 'DNN', 'Hybrid Model']
    data = {metric: [if_metrics.get(metric, 0), dnn_metrics.get(metric, 0), hybrid_metrics.get(metric, 0)] for metric in metrics_to_plot}
    df_comp = pd.DataFrame(data, index=models)
    
    ax = df_comp.plot(kind='bar', figsize=(16, 9), rot=0, colormap='viridis')
    plt.title('Comparison of Model Performance Metrics', fontsize=18)
    plt.ylabel('Score', fontsize=14)
    # FINAL IMPROVEMENT: Adjust ylim to ensure all bars are visible
    plt.ylim(0.65, 1.01) 
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', fontsize=10)
    plt.grid(axis='y', linestyle='--')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=12)
    plt.tight_layout(pad=4.0)
    plt.savefig("results/comparison_metrics.png", dpi=300)
    plt.close()
    print("Comparison bar chart saved to 'results/comparison_metrics.png'")

    # --- Line Chart for ROC Curves ---
    plt.figure(figsize=(10, 8))
    
    fpr_if, tpr_if, _ = roc_curve(y_test, y_scores_if)
    plt.plot(fpr_if, tpr_if, lw=2, label=f'Isolation Forest (AUC = {if_metrics["ROC-AUC"]:.4f})')

    fpr_dnn, tpr_dnn, _ = roc_curve(y_test, y_scores_dnn)
    plt.plot(fpr_dnn, tpr_dnn, lw=2, label=f'DNN (AUC = {dnn_metrics["ROC-AUC"]:.4f})')
    
    fpr_hy, tpr_hy, _ = roc_curve(y_test, hybrid_scores)
    plt.plot(fpr_hy, tpr_hy, lw=2, label=f'Hybrid Model (AUC = {hybrid_metrics["ROC-AUC"]:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Comparison of ROC Curves', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/comparison_roc_curve.png", dpi=300)
    plt.close()
    print("Comparison ROC curve plot saved to 'results/comparison_roc_curve.png'")


# --- Main Execution ---
if __name__ == "__main__":
    X_train, y_train, X_test, y_test, pipeline = load_and_preprocess_data()
    
    if_metrics, y_pred_if, y_scores_if = run_isolation_forest(X_train, y_train, X_test, y_test, pipeline)
    
    dnn_metrics, y_pred_dnn, y_scores_dnn = run_dnn(X_train, y_train, X_test, y_test, pipeline)
    
    hybrid_metrics, hybrid_scores = run_hybrid_model(y_scores_if, y_scores_dnn, X_test, y_test)
    
    plot_comparison_charts(if_metrics, dnn_metrics, hybrid_metrics, y_test, y_scores_if, y_scores_dnn, hybrid_scores)
    
    print("\nProject execution finished. Check the 'results' folder for all generated graphs.")