#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import torch
import mediapipe as mp
import mtcnn
import xml.etree.ElementTree as ET
from ultralytics import YOLO
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import os
import traceback
from datetime import datetime
from sklearn.model_selection import train_test_split
import time
import csv
import matplotlib.pyplot as plt

class PedestrianIntentionPredictor:
    def __init__(self):
        # Load models (ensure model file path is correct)
        self.yolo_model = YOLO('yolov5su.pt')  # optional, only needed if using YOLO for detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.mtcnn_detector = mtcnn.MTCNN()
        self.intention_model = self.create_intention_network()
        # Label encoders
        self.action_encoder = LabelEncoder().fit(['standing', 'walking', 'running'])
        self.cross_encoder = LabelEncoder().fit(['crossing', 'not crossing'])

    def create_intention_network(self):
        class IntentionNet(nn.Module):
            def __init__(self, input_size=126, hidden_size=128):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.bn1 = nn.BatchNorm1d(hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                self.bn2 = nn.BatchNorm1d(hidden_size // 2)
                self.dropout = nn.Dropout(0.5)
                # outputs: action (3 classes) and cross (2 classes)
                self.fc3 = nn.Linear(hidden_size // 2, 3)
                self.fc4 = nn.Linear(hidden_size // 2, 2)
            def forward(self, x):
                x = torch.relu(self.bn1(self.fc1(x)))
                x = self.dropout(x)
                x = torch.relu(self.bn2(self.fc2(x)))
                return self.fc3(x), self.fc4(x)
        return IntentionNet()

    def save_model(self, path):
        """
        Save trained model checkpoint (weights + label classes + meta).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.intention_model.state_dict(),
            "action_classes": list(self.action_encoder.classes_),
            "cross_classes": list(self.cross_encoder.classes_),
            "model_kwargs": {"input_size": 126, "hidden_size": 128},
            "saved_at": datetime.now().isoformat()
        }
        torch.save(checkpoint, path)
        print(f"Model checkpoint saved to: {path}")

    def parse_xml_annotation(self, xml_path):
        """
        Robust XML parsing (handles common malformed cases).
        Returns list of annotations: {'frame', 'id', 'action', 'cross', 'bbox'}
        """
        print(f"Parsing XML annotations from: {xml_path}")

        if not os.path.exists(xml_path):
            print(f"ERROR: XML file does not exist at {xml_path}")
            return []

        try:
            with open(xml_path, 'r') as f:
                xml_content = f.read()
            xml_content = xml_content.lstrip()
            if not xml_content.startswith('<?xml') and not xml_content.startswith('<annotations>'):
                xml_content = f'<annotations>{xml_content}</annotations>'
            root = ET.fromstring(xml_content)
            tracks = []
            if root.tag == 'track':
                tracks.append(root)
            else:
                tracks = root.findall('.//track')
            print(f"Total Tracks Found: {len(tracks)}")
            processed_annotations = []
            for track in tracks:
                track_id = track.get('id', 'unknown')
                boxes = track.findall('box')
                for box in boxes:
                    frame = box.get('frame', 'unknown')
                    xtl = float(box.get('xtl', 0))
                    ytl = float(box.get('ytl', 0))
                    xbr = float(box.get('xbr', 0))
                    ybr = float(box.get('ybr', 0))
                    attributes = {}
                    for attribute in box.findall('attribute'):
                        name = attribute.get('name', '')
                        value = attribute.text
                        attributes[name] = value
                    action = self.map_action(attributes.get('moving status', '__undefined__'))
                    cross = self.map_cross(attributes.get('Cross', 'not crossing'))
                    annotation = {
                        'frame': int(frame),
                        'id': track_id,
                        'action': action,
                        'cross': cross,
                        'bbox': [xtl, ytl, xbr, ybr]
                    }
                    processed_annotations.append(annotation)
            print(f"Total Annotations Extracted: {len(processed_annotations)}")
            return processed_annotations

        except ET.ParseError as e:
            print(f"XML Parsing Error: {e}")
            print("Attempting to fix malformed XML...")
            try:
                corrected_xml = self.fix_malformed_xml(xml_path)
                corrected_xml = corrected_xml.lstrip()
                root = ET.fromstring(corrected_xml)
                tracks = []
                if root.tag == 'track':
                    tracks.append(root)
                else:
                    tracks = root.findall('.//track')
                print(f"After correction - Total Tracks Found: {len(tracks)}")
                processed_annotations = []
                for track in tracks:
                    track_id = track.get('id', 'unknown')
                    boxes = track.findall('box')
                    for box in boxes:
                        frame = box.get('frame', 'unknown')
                        xtl = float(box.get('xtl', 0))
                        ytl = float(box.get('ytl', 0))
                        xbr = float(box.get('xbr', 0))
                        ybr = float(box.get('ybr', 0))
                        attributes = {}
                        for attribute in box.findall('attribute'):
                            name = attribute.get('name', '')
                            value = attribute.text
                            attributes[name] = value
                        action = self.map_action(attributes.get('moving status', '__undefined__'))
                        cross = self.map_cross(attributes.get('Cross', 'not crossing'))
                        annotation = {
                            'frame': int(frame),
                            'id': track_id,
                            'action': action,
                            'cross': cross,
                            'bbox': [xtl, ytl, xbr, ybr]
                        }
                        processed_annotations.append(annotation)
                print(f"After correction - Total Annotations Extracted: {len(processed_annotations)}")
                return processed_annotations
            except Exception as inner_e:
                print(f"Failed to repair XML: {inner_e}")
                traceback.print_exc()
                return []
        except Exception as e:
            print(f"Unexpected Error: {e}")
            traceback.print_exc()
            return []

    def fix_malformed_xml(self, xml_path):
        """
        Attempt a light-weight fix for common malformations.
        This is a safe heuristic — for serious corruption use an XML cleaning tool.
        """
        with open(xml_path, 'r', errors='ignore') as f:
            content = f.read()
        # Simple heuristics:
        content = content.replace('&', '&amp;')
        if not content.strip().startswith('<'):
            content = '<annotations>' + content + '</annotations>'
        return content

    def map_action(self, original_action):
        action_map = {'standing': 'standing', 'walking': 'walking', 'running': 'running', '__undefined__': 'standing'}
        return action_map.get((original_action or '__undefined__').lower(), 'standing')

    def map_cross(self, original_cross):
        cross_map = {'crossing': 'crossing', 'not crossing': 'not crossing'}
        return cross_map.get((original_cross or 'not crossing').lower(), 'not crossing')

    def extract_features(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, min(x1, w-1)), max(0, min(y1, h-1)), max(0, min(x2, w-1)), max(0, min(y2, h-1))
        if x1 >= x2 or y1 >= y2: return None
        ped_img = frame[y1:y2, x1:x2]
        if ped_img.size == 0: return None
        rgb_img = cv2.cvtColor(ped_img, cv2.COLOR_BGR2RGB)
        if rgb_img.shape[0] < 10 or rgb_img.shape[1] < 10: return None
        pose_results = self.pose.process(rgb_img)
        features = []
        if pose_results.pose_landmarks:
            for lm in pose_results.pose_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z, lm.visibility])
        bbox_w, bbox_h = (x2-x1)/w, (y2-y1)/h
        aspect = bbox_w / bbox_h if bbox_h > 0 else 0
        features.extend([bbox_w, bbox_h, aspect])
        center_x, center_y = (x1+x2)/(2*w), (y1+y2)/(2*h)
        features.extend([center_x, center_y])
        while len(features) < 126: features.append(0)
        return np.array(features[:126])

    def extract_all_features(self, video_path, annotations):
        cap = cv2.VideoCapture(video_path)
        features, labels = [], []
        frame_idx = 0
        frame_map = {}
        for ann in annotations:
            frame_map.setdefault(ann['frame'], []).append(ann)
        print("Extracting features...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            anns = frame_map.get(frame_idx, [])
            for ann in anns:
                feat = self.extract_features(frame, ann['bbox'])
                if feat is not None:
                    features.append(feat)
                    labels.append({'action': ann['action'], 'cross': ann['cross']})
            if frame_idx % 100 == 0: print(f"Frames processed: {frame_idx}")
        cap.release()
        print(f"Total features: {len(features)}")
        return np.array(features), labels

    def train_model(self, X, labels, output_dir=".", max_epochs=200, save_every=10, weight_cross=1.5):
        """
        Trains the multi-task intention model.
        Returns (model, metrics_dict). Metrics include detailed training diagnostics.
        """
        y_action = self.action_encoder.transform([l['action'] for l in labels])
        y_cross = self.cross_encoder.transform([l['cross'] for l in labels])

        import collections
        action_counts = collections.Counter(y_action)
        cross_counts = collections.Counter(y_cross)
        print("\nClass Distribution (Action):", action_counts)
        print("Class Distribution (Cross):", cross_counts)

        X_train, X_temp, y_action_train, y_action_temp, y_cross_train, y_cross_temp = train_test_split(
                      X, y_action, y_cross, test_size=0.2, random_state=42, stratify=y_action if len(set(y_action)) > 1 else None)
        X_val, X_test, y_action_val, y_action_test, y_cross_val, y_cross_test = train_test_split(
                    X_temp, y_action_temp, y_cross_temp, test_size=0.5, random_state=42, stratify=y_action_temp if len(set(y_action_temp)) > 1 else None)
        print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
        y_action_train_tensor = torch.tensor(y_action_train, dtype=torch.long)
        y_action_val_tensor = torch.tensor(y_action_val, dtype=torch.long)
        y_action_test_tensor  = torch.tensor(y_action_test, dtype=torch.long)
        y_cross_train_tensor = torch.tensor(y_cross_train, dtype=torch.long)
        y_cross_val_tensor = torch.tensor(y_cross_val, dtype=torch.long)
        y_cross_test_tensor  = torch.tensor(y_cross_test, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.intention_model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        best_val_loss = float('inf')
        patience, patience_counter = 10, 0

        # Diagnostics storage
        train_joint_losses, val_joint_losses = [], []
        train_action_losses, val_action_losses = [], []
        train_cross_losses, val_cross_losses = [], []
        train_action_accs, val_action_accs = [], []
        train_cross_accs, val_cross_accs = [], []
        epochs_ran = 0

        train_start = time.time()
        for epoch in range(max_epochs):
            epochs_ran = epoch + 1
            self.intention_model.train()
            optimizer.zero_grad()
            action_out_train, cross_out_train = self.intention_model(X_train_tensor)
            action_loss_train = criterion(action_out_train, y_action_train_tensor)
            cross_loss_train = criterion(cross_out_train, y_cross_train_tensor)
            joint_loss_train = action_loss_train + weight_cross * cross_loss_train
            joint_loss_train.backward()
            torch.nn.utils.clip_grad_norm_(self.intention_model.parameters(), 1.0)
            optimizer.step()
            train_joint_losses.append(joint_loss_train.item())
            train_action_losses.append(action_loss_train.item())
            train_cross_losses.append(cross_loss_train.item())

            # compute train accuracies
            with torch.no_grad():
                act_pred_train = torch.argmax(action_out_train, dim=1).cpu().numpy()
                cross_pred_train = torch.argmax(cross_out_train, dim=1).cpu().numpy()
                t_action_acc = accuracy_score(y_action_train, act_pred_train) * 100 if len(y_action_train)>0 else 0.0
                t_cross_acc = accuracy_score(y_cross_train, cross_pred_train) * 100 if len(y_cross_train)>0 else 0.0
                train_action_accs.append(t_action_acc)
                train_cross_accs.append(t_cross_acc)

            # Validation step
            self.intention_model.eval()
            with torch.no_grad():
                val_action_out, val_cross_out = self.intention_model(X_val_tensor)
                action_loss_val = criterion(val_action_out, y_action_val_tensor)
                cross_loss_val = criterion(val_cross_out, y_cross_val_tensor)
                joint_loss_val = action_loss_val + weight_cross * cross_loss_val
                val_joint_losses.append(joint_loss_val.item())
                val_action_losses.append(action_loss_val.item())
                val_cross_losses.append(cross_loss_val.item())

                # compute val accuracies
                val_act_pred = torch.argmax(val_action_out, dim=1).cpu().numpy()
                val_cross_pred = torch.argmax(val_cross_out, dim=1).cpu().numpy()
                v_action_acc = accuracy_score(y_action_val, val_act_pred) * 100
                v_cross_acc = accuracy_score(y_cross_val, val_cross_pred) * 100
                val_action_accs.append(v_action_acc)
                val_cross_accs.append(v_cross_acc)

                scheduler.step(joint_loss_val)

                if joint_loss_val < best_val_loss:
                    best_val_loss = joint_loss_val
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.intention_model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        # load best
                        self.intention_model.load_state_dict(best_model_state)
                        break

            if (epoch+1) % save_every == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{max_epochs}: Train Joint Loss={joint_loss_train.item():.4f}, Val Joint Loss={joint_loss_val.item():.4f}")
                print(f"  Train: Action Loss={action_loss_train.item():.4f}, Cross Loss={cross_loss_train.item():.4f}, Aacc={t_action_acc:.2f}%, Cacc={t_cross_acc:.2f}%")
                print(f"  Val:   Action Loss={action_loss_val.item():.4f}, Cross Loss={cross_loss_val.item():.4f}, Aacc={v_action_acc:.2f}%, Cacc={v_cross_acc:.2f}%")

        train_time = time.time() - train_start

        # Final evaluation on validation set
        self.intention_model.eval()
        with torch.no_grad():
            test_action_out, test_cross_out = self.intention_model(X_test_tensor)
            test_act_pred = torch.argmax(test_action_out, dim=1).numpy()
            test_cross_pred = torch.argmax(test_cross_out, dim=1).numpy()

            # Action metrics
            test_action_acc = accuracy_score(y_action_test, test_act_pred) * 100
            test_action_prec = precision_score(y_action_test, test_act_pred, average='weighted', zero_division=0) * 100
            test_action_rec = recall_score(y_action_test, test_act_pred, average='weighted', zero_division=0) * 100
            test_action_f1 = f1_score(y_action_test, test_act_pred, average='weighted', zero_division=0) * 100
            test_action_cm = confusion_matrix(y_action_test, test_act_pred)

            # Cross metrics
            test_cross_acc = accuracy_score(y_cross_test, test_cross_pred) * 100
            test_cross_prec = precision_score(y_cross_test, test_cross_pred, average='weighted', zero_division=0) * 100
            test_cross_rec = recall_score(y_cross_test, test_cross_pred, average='weighted', zero_division=0) * 100
            test_cross_f1 = f1_score(y_cross_test, test_cross_pred, average='weighted', zero_division=0) * 100
            test_cross_cm = confusion_matrix(y_cross_test, test_cross_pred)
            overall_acc = (test_action_acc + test_cross_acc) / 2
            overall_precision = (test_action_prec + test_cross_prec) / 2
            overall_recall = (test_action_rec + test_cross_rec) / 2
            overall_f1 = (test_action_f1 + test_cross_f1) / 2

            # ROC for crossing
            from sklearn.metrics import roc_curve, auc
            cross_probs = torch.softmax(val_cross_out, dim=1).numpy()[:,1]  # prob of 'crossing'
            try:
                fpr, tpr, thresholds = roc_curve(y_cross_val, cross_probs)
                roc_auc = auc(fpr, tpr)
            except Exception as e:
                print("ROC calculation error (maybe single-class in val):", e)
                fpr, tpr, thresholds, roc_auc = None, None, None, None

            # Save ROC plot
            try:
                if roc_auc is not None:
                    plt.figure()
                    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
                    plt.xlabel('False Positive Rate (1 - Specificity)')
                    plt.ylabel('True Positive Rate (Sensitivity)')
                    plt.title('ROC Curve for Crossing Intention')
                    plt.legend(loc="lower right")
                    roc_path = os.path.join(output_dir, "roc_curve_crossing.png")
                    plt.tight_layout()
                    plt.savefig(roc_path, dpi=300)
                    plt.close()
                    print(f"ROC AUC for Crossing = {roc_auc:.4f} (saved to {roc_path})")
                else:
                    print("ROC AUC not available (single-class in validation set).")
            except Exception as e:
                print("Failed to plot/save ROC:", e)

        # Save training diagnostics (CSV)
        csv_path = os.path.join(output_dir, f"training_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['epoch',
                              'train_joint_loss', 'val_joint_loss',
                              'train_action_loss', 'val_action_loss',
                              'train_cross_loss', 'val_cross_loss',
                              'train_action_acc', 'val_action_acc',
                              'train_cross_acc', 'val_cross_acc']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(len(train_joint_losses)):
                    writer.writerow({
                        'epoch': i+1,
                        'train_joint_loss': train_joint_losses[i],
                        'val_joint_loss': val_joint_losses[i] if i < len(val_joint_losses) else '',
                        'train_action_loss': train_action_losses[i],
                        'val_action_loss': val_action_losses[i] if i < len(val_action_losses) else '',
                        'train_cross_loss': train_cross_losses[i],
                        'val_cross_loss': val_cross_losses[i] if i < len(val_cross_losses) else '',
                        'train_action_acc': train_action_accs[i] if i < len(train_action_accs) else '',
                        'val_action_acc': val_action_accs[i] if i < len(val_action_accs) else '',
                        'train_cross_acc': train_cross_accs[i] if i < len(train_cross_accs) else '',
                        'val_cross_acc': val_cross_accs[i] if i < len(val_cross_accs) else ''
                    })
            print(f"Training diagnostics saved to {csv_path}")
        except Exception as e:
            print("Failed to save training diagnostics CSV:", e)

        # Plot losses: joint + individual losses on single plot
        try:
            plt.figure(figsize=(8,5))
            plt.plot(train_joint_losses, label='Train Joint Loss')
            plt.plot(val_joint_losses, label='Val Joint Loss')
            plt.plot(train_action_losses, label='Train Action Loss', linestyle='--')
            plt.plot(val_action_losses, label='Val Action Loss', linestyle='--')
            plt.plot(train_cross_losses, label='Train Cross Loss', linestyle=':')
            plt.plot(val_cross_losses, label='Val Cross Loss', linestyle=':')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Diagnostics - Losses')
            plt.legend()
            plt.tight_layout()
            loss_plot_path = os.path.join(output_dir, f"losses_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(loss_plot_path, dpi=300)
            plt.close()
            print(f"Loss diagnostic plot saved: {loss_plot_path}")
        except Exception as e:
            print("Loss plotting error:", e)

        # Plot accuracies for action & cross
        try:
            plt.figure(figsize=(8,4))
            plt.plot(train_action_accs, label='Train Action Acc')
            plt.plot(val_action_accs, label='Val Action Acc')
            plt.plot(train_cross_accs, label='Train Cross Acc')
            plt.plot(val_cross_accs, label='Val Cross Acc')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Training Diagnostics - Accuracies')
            plt.legend()
            plt.tight_layout()
            acc_plot_path = os.path.join(output_dir, f"accs_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(acc_plot_path, dpi=300)
            plt.close()
            print(f"Accuracy diagnostic plot saved: {acc_plot_path}")
        except Exception as e:
            print("Accuracy plotting error:", e)

        # Print validation summary
        print("\n==== Validation Metrics ====")
        print(f"Action Accuracy: {test_action_acc:.2f}%")
        print(f"Action Precision: {test_action_prec:.2f}%")
        print(f"Action Recall: {test_action_rec:.2f}%")
        print(f"Action F1 Score: {test_action_f1:.2f}%")
        print(f"Action Confusion Matrix:\n{test_action_cm}")
        print(f"Cross Accuracy: {test_cross_acc:.2f}%")
        print(f"Cross Precision: {test_cross_prec:.2f}%")
        print(f"Cross Recall: {test_cross_rec:.2f}%")
        print(f"Cross F1 Score: {test_cross_f1:.2f}%")
        print(f"Cross Confusion Matrix:\n{test_cross_cm}")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        print(f"Training Time: {train_time:.2f} seconds")
        print("===========================")

        metrics = {
            'action': {
                'accuracy': test_action_acc,
                'precision': test_action_prec,
                'recall': test_action_rec,
                'f1': test_action_f1,
                'confusion_matrix': test_action_cm.tolist()
            },
            'cross': {
                'accuracy': test_cross_acc,
                'precision': test_cross_prec,
                'recall': test_cross_rec,
                'f1': test_cross_f1,
                'confusion_matrix': test_cross_cm.tolist()
            },
            'overall_accuracy': overall_acc,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'train_losses': train_joint_losses,
            'val_losses': val_joint_losses,
            'train_action_losses': train_action_losses,
            'val_action_losses': val_action_losses,
            'train_cross_losses': train_cross_losses,
            'val_cross_losses': val_cross_losses,
            'train_action_accs': train_action_accs,
            'val_action_accs': val_action_accs,
            'train_cross_accs': train_cross_accs,
            'val_cross_accs': val_cross_accs,
            'train_time': train_time,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'epochs_ran': epochs_ran,
            'model_framework': "PyTorch, custom MLP with BatchNorm, Dropout, Adam optimizer, ReduceLROnPlateau LR scheduler, early stopping"
        }

        return self.intention_model, metrics

    def predict_intention(self, frame, bbox):
        t_pred_start = time.perf_counter()
        features = self.extract_features(frame, bbox)
        if features is None:
            return None
        X = torch.tensor([features], dtype=torch.float32)
        self.intention_model.eval()
        t_inf_start = time.perf_counter()
        with torch.no_grad():
            action_out, cross_out = self.intention_model(X)
            action_probs = torch.softmax(action_out, dim=1)[0]
            cross_probs = torch.softmax(cross_out, dim=1)[0]
        t_inf_end = time.perf_counter()

        action_pred = torch.argmax(action_probs).item()
        cross_pred = torch.argmax(cross_probs).item()
        action_conf = action_probs[action_pred].item() * 100
        cross_conf = cross_probs[cross_pred].item() * 100
        action_label = self.action_encoder.inverse_transform([action_pred])[0]
        cross_label = self.cross_encoder.inverse_transform([cross_pred])[0]
        if action_conf < 65:
            action_label += '*'
        if cross_conf < 60:
            cross_label += '*'

        t_pred_end = time.perf_counter()
        inference_ms = (t_inf_end - t_inf_start) * 1000.0
        prediction_ms = (t_pred_end - t_pred_start) * 1000.0

        return {
            'action': {'label': action_label, 'confidence': action_conf},
            'cross': {'label': cross_label, 'confidence': cross_conf},
            'timings': {'inference_ms': inference_ms, 'prediction_ms': prediction_ms}
        }

    def generate_output_video(self, video_path, annotations, output_path):
        cap = cv2.VideoCapture(video_path)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        frame_map = {}
        for ann in annotations:
            frame_map.setdefault(ann['frame'], []).append(ann)
        correct_action, correct_cross, total = 0, 0, 0
        total_inference_ms, total_prediction_ms, pred_count = 0.0, 0.0, 0
        frame_idx = 0
        print(f"Saving output video: {output_path}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            anns = frame_map.get(frame_idx, [])
            for ann in anns:
                bbox = ann['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                pred = self.predict_intention(frame, bbox)
                if pred:
                    pa, pc = pred['action']['label'], pred['cross']['label']
                    ta, tc = ann['action'], ann['cross']
                    is_a = pa.replace('*', '') == ta
                    is_c = pc.replace('*', '') == tc
                    correct_action += is_a
                    correct_cross += is_c
                    total += 1
                    # timing aggregation
                    if 'timings' in pred:
                        total_inference_ms += pred['timings'].get('inference_ms', 0.0)
                        total_prediction_ms += pred['timings'].get('prediction_ms', 0.0)
                        pred_count += 1
                    color = (0,255,0) if is_a and is_c else (0,0,255) if not is_a and not is_c else (255,165,0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # overlay predictions and timings
                    cv2.putText(frame, f"A:{pa}({pred['action']['confidence']:.1f}%)", (x1, max(15, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, f"C:{pc}({pred['cross']['confidence']:.1f}%)", (x1, max(35, y1-30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if 'timings' in pred:
                        cv2.putText(
                            frame,
                            f"Inf:{pred['timings']['inference_ms']:.1f}ms Pred:{pred['timings']['prediction_ms']:.1f}ms",
                            (x1, max(55, y1-50)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255, 255, 0),
                            1
                        )
            if total > 0:
                a_acc = (correct_action/total)*100
                c_acc = (correct_cross/total)*100
                o_acc = ((correct_action+correct_cross)/(2*total))*100
                cv2.putText(frame, f"Frame:{frame_idx} A:{a_acc:.1f}% C:{c_acc:.1f}% O:{o_acc:.1f}%", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            out.write(frame)
            if frame_idx % 50 == 0:
                print(f"Frames written: {frame_idx}")
                if pred_count > 0:
                    avg_inf = total_inference_ms / pred_count
                    avg_pred = total_prediction_ms / pred_count
                    print(f"Average Inference Time: {avg_inf:.2f} ms | Average Prediction Time: {avg_pred:.2f} ms (over {pred_count} predictions)")
        cap.release()
        out.release()
        if pred_count > 0:
            avg_inf = total_inference_ms / pred_count
            avg_pred = total_prediction_ms / pred_count
            print(f"Final Average Inference Time: {avg_inf:.2f} ms | Final Average Prediction Time: {avg_pred:.2f} ms (over {pred_count} predictions)")
        print(f"Video saved: {output_path}")

def main():
    start_time = time.time()
    predictor = PedestrianIntentionPredictor()
    # Paths: change these to your dataset paths
    video_path = r"E:\DATASET FINAL FINAL\clip_012.mp4"
    annotation_path = r"D:\sem7\final year project\s8 final year\annotations\clip_012.xml"
    output_dir = r"E:\out_custom_roc_new11"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parse annotations
    annotations = predictor.parse_xml_annotation(annotation_path)
    if not annotations:
        print("No annotations found.")
        return

    # Feature extraction
    X, labels = predictor.extract_all_features(video_path, annotations)
    if len(X) == 0:
        print("No features extracted.")
        return

    # Train model with diagnostics
    print("\nTraining model (with multi-task diagnostics)...")
    model, metrics = predictor.train_model(X, labels, output_dir=output_dir, max_epochs=200, save_every=10, weight_cross=1.5)

    # Save trained model checkpoints
    model_save_path = os.path.join(output_dir, f"intention_model_{timestamp}.pth")
    predictor.save_model(model_save_path)
    # Also save a copy to Downloads for convenience
    downloads_dir = r"E:\out_custom_roc_new9"
    downloads_model_path = os.path.join(downloads_dir, f"intention_model_{timestamp}.pth")
    predictor.save_model(downloads_model_path)

    # Print structured summary
    print("\n========== TRAIN/VAL SPLIT ==========")
    print(f"Train samples: {metrics['train_samples']}")
    print(f"Validation samples: {metrics['val_samples']}")
    print(f"Test samples: {metrics['test_samples']}")
    print(f"Epochs ran: {metrics.get('epochs_ran', 'N/A')}")
    print("=====================================")
    print("\n========== MODEL FRAMEWORK ==========")
    print(metrics['model_framework'])
    print("=====================================")
    print("\n========== METRICS SUMMARY ==========")
    print(f"Action:    Acc={metrics['action']['accuracy']:.2f}%  Prec={metrics['action']['precision']:.2f}%  Recall={metrics['action']['recall']:.2f}%  F1={metrics['action']['f1']:.2f}%")
    print(f"Cross:     Acc={metrics['cross']['accuracy']:.2f}%  Prec={metrics['cross']['precision']:.2f}%  Recall={metrics['cross']['recall']:.2f}%  F1={metrics['cross']['f1']:.2f}%")
    print(f"Overall:   Acc={metrics['overall_accuracy']:.2f}%  Prec={metrics['overall_precision']:.2f}%  Recall={metrics['overall_recall']:.2f}%  F1={metrics['overall_f1']:.2f}%")
    print(f"Training Time: {metrics['train_time']:.2f} seconds")
    print("=====================================")
    print("\nAction Confusion Matrix:")
    print(np.array(metrics['action']['confusion_matrix']))
    print("Cross Confusion Matrix:")
    print(np.array(metrics['cross']['confusion_matrix']))
    print("=====================================")

    # Save loss & accuracy plots already saved by train_model - additionally copy to timestamped files
    # Output video
    output_video_path = os.path.join(output_dir, f"pedestrian_prediction_{timestamp}.mp4")
    predictor.generate_output_video(video_path, annotations, output_video_path)

    # Save metrics text
    metrics_path = os.path.join(output_dir, f"metrics_{timestamp}.txt")
    try:
        with open(metrics_path, "w") as f:
            f.write("Pedestrian Intention Prediction Metrics\n")
            f.write("======================================\n")
            f.write(f"Train samples: {metrics['train_samples']}\n")
            f.write(f"Validation samples: {metrics['val_samples']}\n")
            f.write(f"Test samples: {metrics['test_samples']}\n")
            f.write(f"Epochs ran: {metrics.get('epochs_ran', 'N/A')}\n")
            f.write(f"Model Framework: {metrics['model_framework']}\n")
            f.write("======================================\n")
            f.write(f"Action Accuracy: {metrics['action']['accuracy']:.2f}%\n")
            f.write(f"Action Precision: {metrics['action']['precision']:.2f}%\n")
            f.write(f"Action Recall: {metrics['action']['recall']:.2f}%\n")
            f.write(f"Action F1 Score: {metrics['action']['f1']:.2f}%\n")
            f.write(f"Action Confusion Matrix:\n{np.array(metrics['action']['confusion_matrix'])}\n")
            f.write(f"Cross Accuracy: {metrics['cross']['accuracy']:.2f}%\n")
            f.write(f"Cross Precision: {metrics['cross']['precision']:.2f}%\n")
            f.write(f"Cross Recall: {metrics['cross']['recall']:.2f}%\n")
            f.write(f"Cross F1 Score: {metrics['cross']['f1']:.2f}%\n")
            f.write(f"Cross Confusion Matrix:\n{np.array(metrics['cross']['confusion_matrix'])}\n")
            f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%\n")
            f.write(f"Overall Precision: {metrics['overall_precision']:.2f}%\n")
            f.write(f"Overall Recall: {metrics['overall_recall']:.2f}%\n")
            f.write(f"Overall F1 Score: {metrics['overall_f1']:.2f}%\n")
            f.write(f"Training Time: {metrics['train_time']:.2f} seconds\n")
        print(f"Metrics saved: {metrics_path}")
    except Exception as e:
        print("Failed to save metrics file:", e)

    print(f"Output video: {output_video_path}")
    print("Done.")
    print(f"Total time: {time.time()-start_time:.2f} seconds")

if __name__ == "__main__":
    main()

