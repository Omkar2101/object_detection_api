import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import torch
from tqdm import tqdm
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model on 5 object classes")
    parser.add_argument('--model', type=str, default='runs/train/exp3/weights/best.pt', 
                       help='Path to YOLOv8 model (default: runs/train/exp3/weights/best.pt)')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing class folders')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='Confidence threshold for detection')
    return parser.parse_args()

class YOLOEvaluator:
    def __init__(self, model_path, conf_threshold=0.25, results_dir='results'):
        """
        Initialize the YOLOv8 model evaluator
        
        Args:
            model_path: Path to YOLOv8 model
            conf_threshold: Confidence threshold for detection
            results_dir: Directory to save evaluation results
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load the trained model from runs/train/exp3
        print(f"Loading trained YOLOv8 model from {model_path}...")
        try:
            self.model = YOLO(model_path)
            print("Model loaded successfully!")
            
            # Get the class names from the model
            self.class_names = self.model.names
            print(f"Model classes: {self.class_names}")
            
            # Create class mapping from model class names
            self.class_mapping = {name: idx for idx, name in self.class_names.items()}
            print(f"Class mapping created: {self.class_mapping}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Set confidence threshold
        self.conf_threshold = conf_threshold
        print(f"Detection confidence threshold: {self.conf_threshold}")
        
        # Create results directories
        self.results_dir = results_dir
        self.cm_dir = os.path.join(results_dir, 'confusion_matrices')
        self.metrics_dir = os.path.join(results_dir, 'metrics')
        
        os.makedirs(self.cm_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
    
    def detect_object(self, image_path, target_class_id):
        """
        Detect if a specific object class is present in the image
        
        Args:
            image_path: Path to the image file
            target_class_id: YOLO class ID for the target object
            
        Returns:
            Boolean indicating if the target object was detected
        """
        # Run inference
        results = self.model(image_path, conf=self.conf_threshold)
        
        # Check if the target class is in the results
        detected = False
        
        # Get the predicted classes
        if len(results) > 0:
            result = results[0]  # First image result
            boxes = result.boxes
            
            if len(boxes) > 0:
                # Check if any of the detected objects match the target class
                for box in boxes:
                    cls = int(box.cls[0].item())
                    if cls == target_class_id:
                        detected = True
                        break
        
        return detected
    
    def evaluate_class(self, class_name, positive_folder, negative_folder):
        """
        Evaluate YOLO model performance for a specific class
        
        Args:
            class_name: Name of the class being evaluated
            positive_folder: Folder with images containing the object
            negative_folder: Folder with images not containing the object
        
        Returns:
            Dictionary with performance metrics and prediction lists
        """
        class_id = self.class_mapping.get(class_name)
        if class_id is None:
            raise ValueError(f"Class '{class_name}' not found in class_mapping. Available classes: {list(self.class_mapping.keys())}")
        
        # Lists to store ground truth and predictions
        y_true = []
        y_pred = []
        eval_images = []  # Store image paths for potential error analysis
        
        # Process positive examples
        print(f"Processing positive examples for {class_name} (class ID: {class_id})...")
        pos_files = [f for f in os.listdir(positive_folder) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for img_file in tqdm(pos_files, desc=f"Positive {class_name} images"):
            img_path = os.path.join(positive_folder, img_file)
            prediction = self.detect_object(img_path, class_id)
            y_true.append(True)  # Ground truth is True (object present)
            y_pred.append(prediction)
            eval_images.append((img_path, True, prediction))  # Store for error analysis
        
        # Process negative examples
        print(f"Processing negative examples for {class_name}...")
        neg_files = [f for f in os.listdir(negative_folder) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for img_file in tqdm(neg_files, desc=f"Negative {class_name} images"):
            img_path = os.path.join(negative_folder, img_file)
            prediction = self.detect_object(img_path, class_id)
            y_true.append(False)  # Ground truth is False (object not present)
            y_pred.append(prediction)
            eval_images.append((img_path, False, prediction))  # Store for error analysis
        
        # Calculate metrics
        # Handle edge cases where confusion matrix might fail
        if len(y_true) == 0:
            raise ValueError(f"No images found for class {class_name}")
            
        if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
            # Handle the case where all predictions are the same
            if all(y_pred):  # All predictions are True
                tp = sum(y_true)
                fp = sum(not t for t in y_true)
                tn = 0
                fn = 0
            else:  # All predictions are False
                tp = 0
                fp = 0
                tn = sum(not t for t in y_true)
                fn = sum(y_true)
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0) if sum(y_pred) > 0 else 0
        recall = recall_score(y_true, y_pred, zero_division=0) if sum(y_true) > 0 else 0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Create results dictionary
        results = {
            'class': class_name,
            'class_id': class_id,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'total_images': len(y_true),
            'positive_examples': sum(y_true),
            'negative_examples': len(y_true) - sum(y_true)
        }
        
        # Store error cases for analysis
        error_cases = []
        for img_path, gt, pred in eval_images:
            if (gt and not pred) or (not gt and pred):  # FN or FP
                error_cases.append({
                    'image': img_path,
                    'ground_truth': 'Positive' if gt else 'Negative',
                    'prediction': 'Positive' if pred else 'Negative',
                    'error_type': 'False Negative' if gt and not pred else 'False Positive'
                })
        
        return results, y_true, y_pred, error_cases
    
    def plot_confusion_matrix(self, class_name, tn, fp, fn, tp):
        """Plot confusion matrix for a class"""
        cm = np.array([[tn, fp], [fn, tp]])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {class_name}')
        plt.tight_layout()
        
        save_path = os.path.join(self.cm_dir, f'{class_name}_confusion_matrix.png')
        plt.savefig(save_path)
        plt.close()
        return save_path
    
    def plot_comparative_metrics(self, all_results):
        """Plot comparative metrics for all classes"""
        # Extract data
        classes = [r['class'] for r in all_results]
        precision = [r['precision'] for r in all_results]
        recall = [r['recall'] for r in all_results]
        f1 = [r['f1_score'] for r in all_results]
        accuracy = [r['accuracy'] for r in all_results]
        
        # Plot
        plt.figure(figsize=(12, 8))
        width = 0.2
        x = np.arange(len(classes))
        
        # Create bars
        plt.bar(x - width*1.5, precision, width, label='Precision')
        plt.bar(x - width/2, recall, width, label='Recall')
        plt.bar(x + width/2, f1, width, label='F1 Score')
        plt.bar(x + width*1.5, accuracy, width, label='Accuracy')
        
        # Customize plot
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Comparison of Metrics Across Classes')
        plt.xticks(x, classes)
        plt.legend()
        
        # Add value labels
        for i, v in enumerate(precision):
            plt.text(i - width*1.5, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
        for i, v in enumerate(recall):
            plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
        for i, v in enumerate(f1):
            plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
        for i, v in enumerate(accuracy):
            plt.text(i + width*1.5, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
        
        plt.ylim(0, 1.15)
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.results_dir, 'comparative_metrics.png')
        plt.savefig(save_path)
        plt.close()
        return save_path
    
    def create_report(self, all_results, error_cases_by_class):
        """Create a detailed report of all metrics"""
        # Convert results to DataFrame
        df = pd.DataFrame(all_results)
        
        # Save detailed CSV report
        csv_path = os.path.join(self.metrics_dir, 'detailed_metrics.csv')
        df.to_csv(csv_path, index=False)
        
        # Create error cases report for each class
        for class_name, errors in error_cases_by_class.items():
            if errors:
                error_df = pd.DataFrame(errors)
                error_csv = os.path.join(self.metrics_dir, f'{class_name}_errors.csv')
                error_df.to_csv(error_csv, index=False)
                print(f"Saved {len(errors)} error cases for {class_name} to {error_csv}")
        
        # Create a summary table
        summary = df[['class', 'precision', 'recall', 'f1_score', 'accuracy']].copy()
        
        # Add average row
        avg_row = {
            'class': 'AVERAGE',
            'precision': summary['precision'].mean(),
            'recall': summary['recall'].mean(),
            'f1_score': summary['f1_score'].mean(),
            'accuracy': summary['accuracy'].mean()
        }
        summary = pd.concat([summary, pd.DataFrame([avg_row])], ignore_index=True)
        
        # Save summary CSV
        summary_path = os.path.join(self.metrics_dir, 'summary_metrics.csv')
        summary.to_csv(summary_path, index=False)
        
        print(f"\nSaved detailed report to {csv_path}")
        print(f"Saved summary report to {summary_path}")
        
        return summary
    
    def evaluate_all_classes(self, classes_info):
        """
        Evaluate all classes and generate reports
        
        Args:
            classes_info: List of dictionaries with class info
        """
        all_results = []
        error_cases_by_class = {}
        
        start_time = time.time()
        
        for class_info in classes_info:
            class_name = class_info['name']
            print(f"\n{'='*50}")
            print(f"Evaluating class: {class_name}")
            print(f"{'='*50}")
            
            try:
                results, y_true, y_pred, error_cases = self.evaluate_class(
                    class_name,
                    class_info['positive_folder'],
                    class_info['negative_folder']
                )
                
                all_results.append(results)
                error_cases_by_class[class_name] = error_cases
                
                # Print metrics
                print(f"\nResults for {class_name}:")
                print(f"Total images evaluated: {results['total_images']}")
                print(f"Positive examples: {results['positive_examples']}")
                print(f"Negative examples: {results['negative_examples']}")
                print(f"True Positives: {results['true_positives']}")
                print(f"False Positives: {results['false_positives']}")
                print(f"True Negatives: {results['true_negatives']}")
                print(f"False Negatives: {results['false_negatives']}")
                print(f"Precision: {results['precision']:.4f}")
                print(f"Recall: {results['recall']:.4f}")
                print(f"F1 Score: {results['f1_score']:.4f}")
                print(f"Accuracy: {results['accuracy']:.4f}")
                
                # Calculate False Positive Rate
                if (results['false_positives'] + results['true_negatives']) > 0:
                    fpr = results['false_positives'] / (results['false_positives'] + results['true_negatives'])
                    print(f"False Positive Rate: {fpr:.4f}")
                else:
                    print("False Positive Rate: N/A")
                
                # Plot confusion matrix
                cm_path = self.plot_confusion_matrix(
                    class_name,
                    results['true_negatives'],
                    results['false_positives'],
                    results['false_negatives'],
                    results['true_positives']
                )
                print(f"Saved confusion matrix to {cm_path}")
                
                # Print error analysis
                if error_cases:
                    print(f"\nFound {len(error_cases)} error cases for {class_name}")
                    print(f"False Positives: {sum(1 for e in error_cases if e['error_type'] == 'False Positive')}")
                    print(f"False Negatives: {sum(1 for e in error_cases if e['error_type'] == 'False Negative')}")
            
            except Exception as e:
                print(f"Error evaluating class {class_name}: {e}")
                print("Skipping this class and continuing with others...")
                continue
        
        if not all_results:
            print("No classes were successfully evaluated. Please check your class names and folders.")
            return None
        
        # Create comparative visualization
        comp_path = self.plot_comparative_metrics(all_results)
        print(f"Saved comparative metrics visualization to {comp_path}")
        
        # Create and print summary report
        summary = self.create_report(all_results, error_cases_by_class)
        
        # Print execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nEvaluation completed in {execution_time:.2f} seconds")
        
        print("\nSummary of metrics across all classes:")
        print(summary)
        
        return summary

def verify_image_folder(folder_path):
    """Verify that a folder exists and contains images"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created empty directory: {folder_path}")
        return False
    
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"Warning: No image files found in {folder_path}")
        return False
    
    print(f"Found {len(image_files)} images in {folder_path}")
    return True

def main():
    args = parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        print("Please check that your trained model exists at this location.")
        print("If your model is saved elsewhere, use the --model argument to specify its path.")
        return
    
    # Initialize the evaluator with your trained model
    evaluator = YOLOEvaluator(
        model_path=args.model,
        conf_threshold=args.conf_threshold,
        results_dir=args.results_dir
    )
    
    # Get available classes from the model
    available_classes = list(evaluator.class_mapping.keys())
    print(f"\nDetected classes in your model: {available_classes}")
    
    # Define your 5 classes based on model's available classes
    # Use the first 5 classes from the model, or as many as are available
    eval_classes = available_classes[:5] if len(available_classes) >= 5 else available_classes
    print(f"\nWill evaluate these classes: {eval_classes}")
    
    # Define class folders
    classes_info = []
    for class_name in eval_classes:
        class_info = {
            'name': class_name,
            'positive_folder': os.path.join(args.data_dir, class_name, 'positive'),
            'negative_folder': os.path.join(args.data_dir, class_name, 'negative')
        }
        
        # Check if folders exist and contain images
        pos_valid = verify_image_folder(class_info['positive_folder'])
        neg_valid = verify_image_folder(class_info['negative_folder'])
        
        if pos_valid and neg_valid:
            classes_info.append(class_info)
        else:
            print(f"Warning: Skipping class '{class_name}' due to missing images.")
            print("Make sure you have both positive and negative image folders for each class.")
            print(f"Expected folders: {class_info['positive_folder']} and {class_info['negative_folder']}")
    
    if not classes_info:
        print("\nError: No valid classes found with proper image folders.")
        print("Please check your data directory structure and make sure images are organized as follows:")
        print("data/class_name/positive/ - Images containing the object")
        print("data/class_name/negative/ - Images without the object")
        return
    
    # Run evaluation on all valid classes
    evaluator.evaluate_all_classes(classes_info)

if __name__ == "__main__":
    main()