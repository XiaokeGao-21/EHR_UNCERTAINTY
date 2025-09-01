import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, roc_auc_score
import os
import warnings
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
warnings.filterwarnings("ignore")
# from run_all_tasks import load_features, process_task
from scipy.special import logit
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve as skl_calibration_curve
import pickle
from sklearn.metrics import confusion_matrix
from matplotlib.patches import Patch

base_dir = "/hpc/group/engelhardlab/xg97"  
# base_dir = "D:/DUKE/LAB/Matt/femr_cuda_env"
tasks_path = f"{base_dir}/EHRSHOT_ASSETS/benchmark" 
features_path = f"{base_dir}/EHRSHOT_ASSETS/features/clmbr_features.pkl"
splits_path = f"{base_dir}/EHRSHOT_ASSETS/splits/person_id_map.csv"

def load_features():
    """Load features and create global key mapping"""
    try:
        with open(features_path, "rb") as f:
            clmbr_features = pickle.load(f)
        
        feature_df = pd.DataFrame(
            clmbr_features["data_matrix"],
            columns=[f'feat_{i}' for i in range(clmbr_features["data_matrix"].shape[1])]
        )
        
        # Create key_global mapping
        patient_ids = clmbr_features["patient_ids"]
        times = clmbr_features["labeling_time"]
        
        # Convert times to string format for consistency
        times_str = [pd.Timestamp(t).isoformat() for t in times]
        
        # Create MultiIndex for mapping (patient_id, time) -> row_index
        key_data = list(zip(patient_ids, times_str))
        key_global = pd.Series(range(len(key_data)), index=pd.MultiIndex.from_tuples(key_data))
        
        return feature_df, key_global
    
    except Exception as e:  
        print(f"Error loading features: {e}")
        return None, None

class SimpleNN(nn.Module): #input_size is 786 as given by CLMBR
    def __init__(self, input_size, hidden_size=256, dropout_rate=0.3):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size) # 1 hidden layer
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate) # MC for uncertainty
        self.layer2 = nn.Linear(hidden_size, 1)
        self.sigmoid2 = nn.Sigmoid()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.sigmoid2(x)
        return x

def train_model(model, train_loader, val_loader, device, patience=3, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    model.train()
    total_loss = 0
    batch_count = 0
    epoch = 0
    
    while True:  # loop until early stopping
        epoch += 1
        # print(f"\nEpoch {epoch}")
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            epoch_loss += loss.item()
            batch_count += 1
            
            # print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        #  epoch summary
        avg_epoch_loss = epoch_loss / len(train_loader)
        # print(f"\nEpoch {epoch} Summary:")
        # print(f"Average Epoch Loss: {avg_epoch_loss:.4f}")
        
        # validation after each epoch
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                outputs = model(val_X)
                val_loss += criterion(outputs, val_y.unsqueeze(1)).item()
        
        val_loss /= len(val_loader)
        # print(f"Validation Loss: {val_loss:.4f}")
        
        # early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            # print("New best model saved!")
        else:
            patience_counter += 1
            print(f"no improvement for {patience_counter} epochs")
            if patience_counter >= patience:
                print(f"early stopping triggered after {epoch} epochs")
                break
        
        model.train()
    
    # best model during training/eval
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def process_task(task_name, features_df, key_global, device, dropout_rate=0.3, e_v_thresholds=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]):
    # task data
    with open(os.path.join(tasks_path, task_name, "all_shots_data.json"), "r") as f:        
        all_data = json.load(f)
    # Only process SHOT="-1"
    shot = "-1"
    fold_data = all_data[task_name][shot]["0"]
    
    pid_train = np.array(fold_data["patient_ids_train_k"], dtype=int)
    pid_val   = np.array(fold_data["patient_ids_val_k"], dtype=int)
    times_train = np.array(fold_data["label_times_train_k"])
    times_val   = np.array(fold_data["label_times_val_k"])
    y_train    = np.array(fold_data["label_values_train_k"])
    y_val      = np.array(fold_data["label_values_val_k"])

    def lookup_rows(pids, times):
        idx = key_global.loc[list(zip(pids, times))].values
        assert not np.isnan(idx).any(), "(pid,time) not found in key_global"
        return idx.astype(int)

    train_rows = lookup_rows(pid_train, times_train)
    val_rows   = lookup_rows(pid_val, times_val)
    X_train = features_df.values[train_rows]
    X_val   = features_df.values[val_rows]

    # Get test data for another source 
    X_test, y_test, times_test, pid_test = get_test_data(task_name)
    y_test_bin = np.where(y_test == 0, 0, 1)
    y_test = y_test_bin

    #########################################################
    # Load age information
    age_path = f"{base_dir}/EHRSHOT_ASSETS/patient_age_from_metadata.csv"
    age_info_df = pd.read_csv(age_path)

    age_info_df['patient_id'] = age_info_df['patient_id'].astype(int)
    
    # Create patient_id to age mapping
    patient_age_map = dict(zip(age_info_df['patient_id'], age_info_df['age_group']))
    
    X_all = np.vstack([X_train, X_val, X_test])  # shape: (N_total, n_feat)
    y_all = np.concatenate([y_train, y_val, y_test])
    time_all = np.concatenate([times_train, times_val, times_test])
    
    # Get patient IDs for all samples
    pid_all = np.concatenate([pid_train, pid_val, pid_test])

    feature_cols = [f'feat_{i}' for i in range(X_all.shape[1])]
    df_all = pd.DataFrame(X_all, columns=feature_cols)
    df_all['label'] = y_all
    df_all['time'] = time_all
    df_all['patient_id'] = pid_all

    # Add age group information
    df_all['age_group'] = df_all['patient_id'].map(patient_age_map)
    # Normalize age group labels to canonical categories
    df_all['age_group'] = df_all['age_group'].astype(str).str.strip().str.lower()
    df_all['age_group'] = df_all['age_group'].replace({
        'middle-aged': 'middle',
        'middle aged': 'middle',
        'elderly': 'elderly',
        'senior': 'senior',
        'young': 'young'
    })
    # Fill unknown age groups with 'middle' (default)
    df_all['age_group'] = df_all['age_group'].replace({'nan': np.nan}).fillna('middle')
    
    print(f"Age distribution in Age-split {task_name}:")
    print(df_all['age_group'].value_counts())
    
    # Split by age groups
    # Test: >60 years (elderly + senior)
    # Train/Val: <60 years (young + middle) - 80% train, 20% val
    test_mask = df_all['age_group'].isin(['elderly', 'senior'])
    train_val_mask = df_all['age_group'].isin(['young', 'middle'])
    
    print(f"Age-based split: >60 years (test): {int(test_mask.sum())}, <60 years (train+val): {int(train_val_mask.sum())}")
    
    df_test_new = df_all[test_mask]
    df_train_val = df_all[train_val_mask]
    if len(df_test_new) == 0 or len(df_train_val) == 0:
        raise ValueError(f"Empty split for task Age-split {task_name}: test={len(df_test_new)}, train+val={len(df_train_val)}. Check age_group mapping.")
    
    # Split train_val into 80% train, 20% val
    n_train_val = len(df_train_val)
    n_train = int(0.8 * n_train_val)
    n_val = n_train_val - n_train
    
    # Random shuffle for train/val split
    df_train_val_shuffled = df_train_val.sample(frac=1, random_state=42).reset_index(drop=True)
    df_train_new = df_train_val_shuffled.iloc[:n_train]
    df_val_new = df_train_val_shuffled.iloc[n_train:]
    
    print(f"Train samples: {len(df_train_new)} (<60 years, 80%)")
    print(f"Val samples: {len(df_val_new)} (<60 years, 20%)")
    print(f"Test samples: {len(df_test_new)} (>60 years)")

    feature_cols = [col for col in df_train_new.columns if col.startswith('feat_')]

    X_train = df_train_new[feature_cols].values
    y_train = df_train_new['label'].values

    X_val = df_val_new[feature_cols].values
    y_val = df_val_new['label'].values

    X_test = df_test_new[feature_cols].values
    y_test = df_test_new['label'].values


    #########################################################

    # normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # tensors and data loaders
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)

    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=256)

    # init and train 
    model = SimpleNN(input_size=X_train.shape[1], dropout_rate=dropout_rate).to(device)
    model = train_model(model, train_loader, val_loader, device, lr=0.001)
    
    # MC Dropout
    n_mc = 300
    
    def compute_uncertainties_mc_dropout(model, x_tensor, mc_passes=300):
        model.train()  # keep dropout active
        probs_mc = []

        with torch.no_grad():
            for _ in range(mc_passes):
                preds = model(x_tensor)
                probs_mc.append(preds.cpu().numpy())  # (N, 1)

        probs_mc = np.stack(probs_mc, axis=1)  # (N, mc_passes, 1) 
        probs_mc = np.squeeze(probs_mc, axis=2)  # (N, mc_passes)
        mean_probs = probs_mc.mean(axis=1)  # (N,)

        # total entropy - H[p] = -p*log(p) - (1-p)*log(1-p)
        predictive_entropy = - (mean_probs * np.log(mean_probs + 1e-8) + 
                                (1 - mean_probs) * np.log(1 - mean_probs + 1e-8))

        # aleatoric entropy: E_p[H(p)] -> mean(H(p))
        entropies = - (probs_mc * np.log(probs_mc + 1e-8) + 
                       (1 - probs_mc) * np.log(1 - probs_mc + 1e-8))  # (N, mc_passes)
        aleatoric_entropy = entropies.mean(axis=1)

        # epistemic uncertainty: predictive - aleatoric
        epistemic_uncertainty = predictive_entropy - aleatoric_entropy

        return mean_probs, predictive_entropy, aleatoric_entropy, epistemic_uncertainty, probs_mc

    # all uncertainty (test set)
    y_proba_mean, predictive_entropy, aleatoric_entropy, epistemic_uncertainty, y_proba_mc = \
        compute_uncertainties_mc_dropout(model, X_test_tensor.to(device), mc_passes=n_mc)
    
    # dynamic prob_threshold by task    
    prob_threshold = np.mean(y_proba_mean)
    print(f"Task Age-split {task_name}: prob_threshold = {prob_threshold:.4f} (mean of all probabilities)")
    
    y_pred = (y_proba_mean > prob_threshold).astype(int)

    # validation set uncertainty
    val_proba_mean, val_predictive_entropy, val_aleatoric_entropy, val_epistemic_uncertainty, _ = \
        compute_uncertainties_mc_dropout(model, X_val_tensor.to(device), mc_passes=n_mc)
    
    # get val predictions 
    # roc_auc, acc, brier, ece, mce, precision, recall, f1, specificity
    def get_metrics(y_true, y_proba, y_pred, n_bins=10):
            y_true_bin = np.where(y_true == 0, 0, 1)
            roc_auc = roc_auc_score(y_true_bin, y_proba) if len(np.unique(y_true_bin)) > 1 else np.nan
            acc = np.mean(y_pred == y_true_bin)
            brier = brier_score_loss(y_true_bin, y_proba)
            ece, mce = compute_ece_mce(y_true, y_proba, n_bins=n_bins)
            
            # Calculate precision, recall, f1-score, and specificity
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
            
            # Handle edge cases where there might be only one class
            if len(np.unique(y_true_bin)) == 1:
                precision = np.nan
                recall = np.nan
                f1 = np.nan
                specificity = np.nan
                sensitivity = np.nan
            else:
                precision = precision_score(y_true_bin, y_pred, zero_division=0)
                recall = recall_score(y_true_bin, y_pred, zero_division=0) # sensitivity = true positive rate
                f1 = f1_score(y_true_bin, y_pred, zero_division=0)
                
                # Calculate specificity (True Negative Rate)
                cm = confusion_matrix(y_true_bin, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                else:
                    specificity = np.nan
                    sensitivity = np.nan
            
            return {
                'roc_auc': roc_auc,
                'accuracy': acc,
                'brier_score': brier,
                'ece': ece,
                'mce': mce,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'sensitivity': sensitivity,
                'n': len(y_true)
            }
    
    ## 95% percentile of val_epistemic_uncertainty
    # e_v = np.percentile(val_epistemic_uncertainty, 95)
    # Use a fixed threshold for all tasks.
    e_v = 0.011

    # split test set into low and high uncertainty
    test_unc = epistemic_uncertainty
    low_unc_mask = test_unc < e_v
    high_unc_mask = test_unc >= e_v

    n_test = len(y_test)


    
    metrics_by_e_v = []
    for threshold in e_v_thresholds:
        # Keep samples with epistemic uncertainty < threshold (low uncertainty)
        keep_idx = np.where(epistemic_uncertainty < threshold)[0]
        y_test_keep = y_test[keep_idx]
        y_proba_mean_keep = y_proba_mean[keep_idx] # [N_keep,]
        y_pred_keep = y_pred[keep_idx] # [N_keep,] 0/1 final predictions
        y_test_keep_bin = np.where(y_test_keep == 0, 0, 1)
        roc_auc = roc_auc_score(y_test_keep_bin, y_proba_mean_keep)
        acc = np.mean(y_pred_keep == y_test_keep_bin)
        brier = brier_score_loss(y_test_keep_bin, y_proba_mean_keep)
        ece, mce = compute_ece_mce(y_test_keep, y_proba_mean_keep, n_bins=10)
        
        # Calculate precision, recall, f1-score, and specificity for the kept test set
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        if len(np.unique(y_test_keep_bin)) == 1:
            precision = np.nan
            recall = np.nan # No positive class present
            f1 = np.nan
            specificity = np.nan
            sensitivity = np.nan
        else:
            precision = precision_score(y_test_keep_bin, y_pred_keep, zero_division=0)
            # sensitivity = true positive rate
            recall = recall_score(y_test_keep_bin, y_pred_keep, zero_division=0) 
            f1 = f1_score(y_test_keep_bin, y_pred_keep, zero_division=0)
            
            # Calculate specificity (True Negative Rate)
            cm = confusion_matrix(y_test_keep_bin, y_pred_keep)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 # true negative rate
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            else:
                specificity = np.nan
                sensitivity = np.nan

        # Full MC predictions and labels for the kept samples
        y_proba_mc_keep = y_proba_mc[keep_idx] # [N_keep, n_mc]
        y_pred_keep = y_pred[keep_idx] # 0/1 final predictions
        y_proba_mean_keep = y_proba_mean_keep # mean probabilities [N_keep,]
        y_test_keep = y_test_keep 

        # low uncertainty
        low_metrics = get_metrics(
            y_test[low_unc_mask],
            y_proba_mean[low_unc_mask],
            y_pred[low_unc_mask]
        )
        # high uncertainty
        high_metrics = get_metrics(
            y_test[high_unc_mask],
            y_proba_mean[high_unc_mask],
            y_pred[high_unc_mask]
        )

        metrics_by_e_v.append({
            'n_train': len(y_train),
            'n_val': len(y_val),
            'n_test': len(y_test),
            'e_v_threshold': threshold,
            'n_keep': len(keep_idx),
            'abstain_rate': 1 - len(keep_idx) / n_test,
            'roc_auc': roc_auc,
            'accuracy': acc,
            'brier_score': brier,
            'ece': ece,
            'mce': mce,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'predictive_entropy_mean': predictive_entropy[keep_idx].mean(),
            'aleatoric_entropy_mean': aleatoric_entropy[keep_idx].mean(),
            'epistemic_uncertainty_mean': epistemic_uncertainty[keep_idx].mean(),
            'epistemic_uncertainty': epistemic_uncertainty[keep_idx],
            'aleatoric_entropy': aleatoric_entropy[keep_idx],
            'predictive_entropy': predictive_entropy[keep_idx],  
            'y_pred': y_pred_keep,
            'y_test_keep': y_test_keep,
            'y_proba_mean_keep': y_proba_mean_keep,
            'y_proba_mc': y_proba_mc_keep,
        })

    # Save per-sample predictions and uncertainties for age-based split (val <60, test >60)
    # preds_dir = os.path.join(base, 'predictions_age')
    df_val_pred = pd.DataFrame({
        'split': 'val',
        'patient_id': df_val_new['patient_id'].values,
        'time': df_val_new['time'].values,
        'age_group': df_val_new['age_group'].values,
        'label': y_val,
        'proba_mean': val_proba_mean,
        'predictive_entropy': val_predictive_entropy,
        'aleatoric_entropy': val_aleatoric_entropy,
        'epistemic_uncertainty': val_epistemic_uncertainty,
    })
    df_test_pred = pd.DataFrame({
        'split': 'test',
        'patient_id': df_test_new['patient_id'].values,
        'time': df_test_new['time'].values,
        'age_group': df_test_new['age_group'].values,
        'label': y_test,
        'proba_mean': y_proba_mean,
        'predictive_entropy': predictive_entropy,
        'aleatoric_entropy': aleatoric_entropy,
        'epistemic_uncertainty': epistemic_uncertainty,
    })
    df_preds = pd.concat([df_val_pred, df_test_pred], ignore_index=True)
    df_preds.insert(0, 'task_name', task_name)
    # out_pred_csv = os.path.join(preds_dir, f'Age-split {task_name}_predictions_age.csv')
    # df_preds.to_csv(out_pred_csv, index=False)

    # New： epistemic_uncertainty and predictive_entropy of  val/test 
    return metrics_by_e_v, {
        'val_epistemic_uncertainty': val_epistemic_uncertainty,
        'val_predictive_entropy': val_predictive_entropy,
        'test_epistemic_uncertainty': epistemic_uncertainty,
        'test_predictive_entropy': predictive_entropy,
        'val_epistemic_uncertainty_95': e_v,
        'test_low_uncertainty_metrics': low_metrics,
        'test_high_uncertainty_metrics': high_metrics,
        'y_proba_mc_full': y_proba_mc,  # Full 300 MC predictions for all test samples
        'y_test_full': y_test,  # Full test labels
        'test_uncertainty_mask': low_unc_mask,  # Low uncertainty mask for coloring
    }

def get_test_data(task_name):
    id_map = pd.read_csv(splits_path)
    test_ids = set(id_map[id_map['split'] == 'test']['omop_person_id'])
    # print(os.path.join('tasks', task_name, 'labeled_patients.csv'))
    
    # labeled_df = pd.read_csv(f'D:/DUKE/LAB/Matt/femr_cuda_env/EHRSHOT_ASSETS/benchmark/Age-split {task_name}/labeled_patients.csv')
    labeled_df = pd.read_csv(os.path.join(tasks_path, task_name, "labeled_patients.csv"))
    test_labeled= labeled_df[labeled_df['patient_id'].isin(test_ids)].copy()
    
    with open(features_path, "rb") as f:
        clmbr_features = pickle.load(f)
    feature_df = pd.DataFrame(
        clmbr_features["data_matrix"],
        columns=[f'feat_{i}' for i in range(clmbr_features["data_matrix"].shape[1])]
    )
    feature_df['patient_id'] = clmbr_features["patient_ids"]
    feature_df['time'] = [pd.Timestamp(t).isoformat() for t in clmbr_features["labeling_time"]]

   
    test_labeled['time'] = pd.to_datetime(test_labeled['prediction_time']).apply(lambda x: x.isoformat())
    merged = test_labeled.merge(feature_df, on=['patient_id', 'time'], how='left')


    feature_cols = [col for col in merged.columns if col.startswith('feat_')]
    X_test = merged[feature_cols].values
    y_test = merged['value'].values

    return X_test, y_test, merged['time'], merged['patient_id']

def calculate_uauc_with_ci_width(y_true, y_proba_mc, ci_width):
    """Calculate uAUC and abstention rate for given confidence interval width"""
    if ci_width == 0:
        pred_risk_lower = pred_risk_upper = np.mean(y_proba_mc, axis=1)
    else:
        lower_percentile = (1 - ci_width) / 2 * 100
        upper_percentile = (1 + ci_width) / 2 * 100
        pred_risk_lower = np.percentile(y_proba_mc, lower_percentile, axis=1)
        pred_risk_upper = np.percentile(y_proba_mc, upper_percentile, axis=1)
    
    y_true = np.array(y_true).astype(int)
    
    # Calculate all sample pairs (negative vs positive)
    all_pairs = y_true[:, np.newaxis] < y_true[np.newaxis, :]
    
    # Decisive pairs
    correct_pairs = all_pairs & (
        pred_risk_upper[:, np.newaxis] < pred_risk_lower[np.newaxis, :]
    )
    incorrect_pairs = all_pairs & (
        pred_risk_lower[:, np.newaxis] > pred_risk_upper[np.newaxis, :]
    )
    
    # Uncertain pairs - overlapping intervals
    uncertain_pairs = all_pairs & ~correct_pairs & ~incorrect_pairs
    
    # Count statistics
    total_pairs = np.sum(all_pairs)
    correct_count = np.sum(correct_pairs)
    incorrect_count = np.sum(incorrect_pairs)
    uncertain_count = np.sum(uncertain_pairs)
    decisive_count = correct_count + incorrect_count
    
    # Calculate uAUC
    uauc = correct_count / decisive_count if decisive_count > 0 else np.nan
    
    # Calculate abstention rate
    ar_uauc = uncertain_count / total_pairs if total_pairs > 0 else 0
    
    return uauc, ar_uauc

def calculate_comprehensive_confusion_metrics_with_ci_width(y_true, y_proba_mc, ci_width, threshold):
    """
    Calculate comprehensive confusion matrix metrics with proper decomposition
    On confident predictions only (outside the CI threshold)
    """
    y_proba_mean = np.mean(y_proba_mc, axis=1)

    if ci_width == 0:
        pred_risk_lower = pred_risk_upper = y_proba_mean
    else:
        lower_percentile = (1 - ci_width) / 2 * 100
        upper_percentile = (1 + ci_width) / 2 * 100
        pred_risk_lower = np.percentile(y_proba_mc, lower_percentile, axis=1)
        pred_risk_upper = np.percentile(y_proba_mc, upper_percentile, axis=1)
    
    y_true = np.array(y_true).astype(int)
    
    # Determine confident predictions
    confident_positive = pred_risk_lower > threshold
    confident_negative = pred_risk_upper < threshold
    confident_mask = confident_positive | confident_negative
    abstain_mask = ~confident_mask

    # ROC-AUC on confident samples
    y_true_conf = y_true[confident_mask]
    y_proba_conf = y_proba_mean[confident_mask]
    roc_auc = np.nan
    if len(np.unique(y_true_conf)) >= 2:
        roc_auc = roc_auc_score(y_true_conf, y_proba_conf)

    # Split by true labels
    positive_mask = y_true == 1
    negative_mask = y_true == 0

    # Abstention stats
    total_positives = np.sum(positive_mask)
    total_negatives = np.sum(negative_mask)
    abstained_true_positives = np.sum(abstain_mask & positive_mask)
    abstained_true_negatives = np.sum(abstain_mask & negative_mask)
    positive_abstain_rate = abstained_true_positives / total_positives if total_positives > 0 else 0
    negative_abstain_rate = abstained_true_negatives / total_negatives if total_negatives > 0 else 0
    overall_abstain_rate = np.sum(abstain_mask) / len(y_true)

    # If no confident samples
    if np.sum(confident_mask) == 0:
        return {
            "roc_auc_confident": roc_auc,
            "tpr": np.nan, "tnr": np.nan,
            "fnr": np.nan, "fpr": np.nan,
            "positive_abstain_rate": positive_abstain_rate,
            "negative_abstain_rate": negative_abstain_rate,
            "overall_abstain_rate": overall_abstain_rate,
            "n_confident": 0,
            "abstained_true_positives": abstained_true_positives,
            "abstained_true_negatives": abstained_true_negatives
        }

    # Make predictions only for confident samples
    confident_preds = np.where(confident_positive[confident_mask], 1, 0)
    confident_true = y_true[confident_mask]

    if len(np.unique(confident_true)) < 2:
        # Handle single-class case
        tn = fp = fn = tp = 0
        if np.all(confident_true == 1):  # all positive
            tp = np.sum(confident_preds)
            fn = len(confident_true) - tp
        else:  # all negative
            tn = np.sum(confident_preds == 0)
            fp = np.sum(confident_preds == 1)
    else:
        tn, fp, fn, tp = confusion_matrix(confident_true, confident_preds, labels=[0, 1]).ravel()

    # Traditional metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    fnr = fn / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "roc_auc_confident": roc_auc,
        "tpr": tpr,
        "tnr": tnr,
        "fnr": fnr,
        "fpr": fpr,
        "positive_abstain_rate": positive_abstain_rate,
        "negative_abstain_rate": negative_abstain_rate,
        "overall_abstain_rate": overall_abstain_rate,
        "n_confident": len(confident_true),
        "abstained_true_positives": abstained_true_positives,
        "abstained_true_negatives": abstained_true_negatives
    }

def analyze_ci_width_effects(task_name, predictions_dict):
    """Analyze the effects of different confidence interval widths"""
    y_true = predictions_dict['y_test']
    y_proba_mc = predictions_dict['y_proba_mc']
    
    # Calculate mean_labels threshold
    mean_labels = np.mean(y_true)
    y_proba_mean = np.mean(y_proba_mc, axis=1)
    prob_threshold = np.mean(y_proba_mean)   # 预测概率的均值

    
    print(f"Task: Age-split {task_name}, Mean labels: {mean_labels:.3f}, Probability threshold: {prob_threshold:.3f}")
    
    # Confidence interval widths to analyze
    ci_widths = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    # Store results
    results = {
        'n_test': [],
        'prob_threshold': prob_threshold,
        'roc_auc_confident':[],
        'uauc': [], 'ar_uauc': [],
        'tpr': [], 'tnr': [], 
        'fnr': [], 'fpr': [],
        'positive_abstain_rate': [], 'negative_abstain_rate': [], 'overall_abstain_rate': [],
        'n_confident':[],
        "abstained_true_positives": [],
        "abstained_true_negatives": []
    }
    
    for ci_width in ci_widths:
        # Calculate uAUC and its abstention rate
        uauc, ar_uauc = calculate_uauc_with_ci_width(y_true, y_proba_mc, ci_width)
        
        # Calculate comprehensive confusion matrix metrics
        confusion_metrics = calculate_comprehensive_confusion_metrics_with_ci_width(
            y_true, y_proba_mc, ci_width, prob_threshold
        )
        
        # Store results
        results['n_test'].append(len(y_true))
        results['roc_auc_confident'].append(confusion_metrics['roc_auc_confident'])
        results['n_confident'].append(confusion_metrics['n_confident'])
        results['uauc'].append(uauc)
        results['ar_uauc'].append(ar_uauc)
        results['tpr'].append(confusion_metrics['tpr'])
        results['tnr'].append(confusion_metrics['tnr'])
        results['fnr'].append(confusion_metrics['fnr'])
        results['fpr'].append(confusion_metrics['fpr'])
        results["abstained_true_positives"].append(confusion_metrics["abstained_true_positives"])
        results["abstained_true_negatives"].append(confusion_metrics["abstained_true_negatives"])
        results['positive_abstain_rate'].append(confusion_metrics['positive_abstain_rate'])
        results['negative_abstain_rate'].append(confusion_metrics['negative_abstain_rate'])
        results['overall_abstain_rate'].append(confusion_metrics['overall_abstain_rate'])
    
    return {
        'ci_widths': ci_widths,
        # 'mean_labels_value': mean_labels, # threshold
        'prob_threshold': prob_threshold, # threshold based on mean of predicted probabilities
        'results': results
    }

def create_improved_ci_width_plots(task_name, task_data, save_dir):
    """Create improved CI width analysis plots
    Plot1: uAUC and Abstention Rate 
    AR = Abstained / Total

    Plot2: Positive Cases Analysis with Proper Decomposition
    AR = Abstained_Pos / Total_Pos

    Plot3: Negative Cases Analysis with Proper Decomposition
    AR = Abstained_Neg / Total_Neg

    """
    ci_widths = task_data['ci_widths']
    results = task_data['results']
    prob_threshold = task_data['prob_threshold']
    # mean_labels_value = task_data['mean_labels_value']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # fig.suptitle(f'Age-split {task_name} - CI Width Analysis (Threshold: Mean Labels = {mean_labels_value:.3f})', fontsize=16)
    fig.suptitle(f'Age-split {task_name} - CI Width Analysis (Probability threshold: {prob_threshold:.3f})', fontsize=16)
    # Plot 1: uAUC and Abstention Rate (clean style like Plot 2)
    ax = axes[0]

    # uAUC
    line1 = ax.plot(
        ci_widths, results['uauc'], 'b-o',
        linewidth=2, markersize=8,
        label='uAUC', markeredgecolor='darkblue', markeredgewidth=1
    )

    # Abstention Rate
    line2 = ax.plot(
        ci_widths, results['ar_uauc'], 'r-s',
        linewidth=2, markersize=8,
        label='Abstention Rate', markeredgecolor='darkred', markeredgewidth=1
    )

    ax.set_xlabel('CI Width', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.set_title('uAUC & Abstention Rate\n(Same Y-axis)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    for i, x in enumerate(ci_widths):
        ax.annotate(f"{results['uauc'][i]:.3f}",
                    (x, results['uauc'][i]),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, color='blue')
        ax.annotate(f"{results['ar_uauc'][i]:.3f}",
                    (x, results['ar_uauc'][i]),
                    textcoords="offset points", xytext=(0, -15),
                    ha='center', fontsize=9, color='red')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=10, framealpha=0.9)
    
    # Plot 2: Positive Cases Analysis with Proper Decomposition
    ax2 = axes[1]
    ax2.plot(ci_widths, results['tpr'], 'g-o', linewidth=3, markersize=8, 
             label='TPR = TP / TP + FN')
    ax2.plot(ci_widths, results['fnr'], 'orange', linestyle='--', marker='^', linewidth=3, markersize=8, 
             label='FNR = FN / TP + FN')
    ax2.plot(ci_widths, results['positive_abstain_rate'], 'r-s', linewidth=3, markersize=8, 
             label='AR = Abstained_Pos / Total_Pos')
    
    ax2.set_xlabel('CI Width', fontsize=12)
    ax2.set_ylabel('Rate', fontsize=12)
    ax2.set_title('Positive Cases Decomposition', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Add value annotations
    for i, x in enumerate(ci_widths):
        ax2.annotate(f"{results['tpr'][i]:.3f}", (x, results['tpr'][i]), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        ax2.annotate(f"{results['positive_abstain_rate'][i]:.3f}", 
                    (x, results['positive_abstain_rate'][i]), 
                    textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9)
    
    # Plot 3: Negative Cases Analysis with Proper Decomposition
    ax3 = axes[2]
    ax3.plot(ci_widths, results['tnr'], 'b-o', linewidth=3, markersize=8, 
             label='TNR = TN / TN + FP')
    ax3.plot(ci_widths, results['fpr'], 'purple', linestyle='--', marker='^', linewidth=3, markersize=8, 
             label='FPR = FP / TN + FP')
    ax3.plot(ci_widths, results['negative_abstain_rate'], 'r-s', linewidth=3, markersize=8, 
             label='AR = Abstained_Neg / Total_Neg')
    
    ax3.set_xlabel('CI Width', fontsize=12)
    ax3.set_ylabel('Rate', fontsize=12)
    ax3.set_title('Negative Cases Decomposition', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Add value annotations
    for i, x in enumerate(ci_widths):
        ax3.annotate(f"{results['tnr'][i]:.3f}", (x, results['tnr'][i]), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        ax3.annotate(f"{results['negative_abstain_rate'][i]:.3f}", 
                    (x, results['negative_abstain_rate'][i]), 
                    textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = f"{save_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(f'{plots_dir}/Age-split {task_name}_improved_ci_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Improved CI width analysis plot saved for Age-split {task_name}")

def save_improved_results_to_excel(all_results, save_dir):
    """Save improved results to Excel with comprehensive and traditional metrics"""
    print("Saving improved results to Excel file...")

    # excel_path = f"{save_dir}/Year_split_ci_width_analysis_results.xlsx"
    csv_path = f"{save_dir}/Age_split_ci_width_analysis_results.csv"

    # with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

        # Sheet 1: Comprehensive results with both metric types
    print("  Creating Sheet 1: Comprehensive Results")
    comprehensive_data = []

    for task_name, task_data in all_results.items():
        ci_widths = task_data['ci_widths']
        prob_threshold = task_data['prob_threshold']
        results = task_data['results']

        for i, ci_width in enumerate(ci_widths):
            row = {
                'task_name': task_name,
                'threshold_mean_preds': prob_threshold,
                'ci_width': ci_width,
                'n_test': results['n_test'][i],
                'n_confident': results['n_confident'][i],
                'uauc': results['uauc'][i],
                'roc_auc_confident': results['roc_auc_confident'][i],
                'ar_uauc': results['ar_uauc'][i],
                'tpr': results['tpr'][i],
                'tnr': results['tnr'][i],
                'fnr': results['fnr'][i],
                'fpr': results['fpr'][i],
                'positive_abstain_rate': results['positive_abstain_rate'][i],
                'negative_abstain_rate': results['negative_abstain_rate'][i],
                'overall_abstain_rate': results['overall_abstain_rate'][i],
                'abstained_true_positives': results['abstained_true_positives'][i],
                'abstained_true_negatives': results['abstained_true_negatives'][i]
            }
            comprehensive_data.append(row)

    comprehensive_df = pd.DataFrame(comprehensive_data)
    comprehensive_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")


def calibration_curve(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    bin_sums = np.zeros(n_bins)
    bin_total = np.zeros(n_bins)
    bin_true = np.zeros(n_bins)
    for i in range(n_bins):
        bin_total[i] = np.sum(binids == i)
        if bin_total[i] > 0:
            bin_sums[i] = np.sum(y_prob[binids == i])
            bin_true[i] = np.sum(y_true[binids == i])
    prob_pred = bin_sums / np.maximum(bin_total, 1)
    prob_true = bin_true / np.maximum(bin_total, 1)
    return prob_pred, prob_true, bin_total

def compute_ece_mce(y_true, y_prob, n_bins=10):
    prob_pred, prob_true, bin_total = calibration_curve(y_true, y_prob, n_bins)
    ece = np.sum(bin_total * np.abs(prob_pred - prob_true)) / np.sum(bin_total)
    mce = np.max(np.abs(prob_pred - prob_true))
    return ece, mce

def create_ci_whisker_boxplot(task_name, 
                              y_proba_mc_full, 
                              y_test_full, 
                              ci_width=0.9,
                              max_samples=100,
                              save_path='results/boxplots_ci_whisker',
                              figsize=(15, 8),
                              dpi=300):
    """
    Create boxplot with CI-based whiskers instead of traditional IQR whiskers.
    This makes confident/uncertain classification visually intuitive.
    """
    
    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    
    # Calculate threshold as mean of all predictions
    # mean_labels = np.mean(y_test_full)  # 使用标签平均值
    # prob_threshold = mean_labels
    prob_threshold = np.mean(y_proba_mc_full)  # use mean of predicted probabilities
    
    # Calculate confidence intervals
    lower_percentile = (1 - ci_width) / 2 * 100
    upper_percentile = (1 + ci_width) / 2 * 100
    pred_risk_lower = np.percentile(y_proba_mc_full, lower_percentile, axis=1)
    pred_risk_upper = np.percentile(y_proba_mc_full, upper_percentile, axis=1)
    y_proba_mean = y_proba_mc_full.mean(axis=1)
    
    # Determine confident predictions based on CI
    confident_positive = pred_risk_lower > prob_threshold
    confident_negative = pred_risk_upper < prob_threshold
    confident_mask = confident_positive | confident_negative
    
    # Sort samples by mean probability
    n_samples = y_proba_mc_full.shape[0]
    sorted_indices = np.argsort(y_proba_mean)
    
    # Sample selection for display
    if n_samples > max_samples:
        sample_indices = np.linspace(0, n_samples - 1, max_samples, dtype=int)
        sorted_sample_indices = sorted_indices[sample_indices]
    else:
        sorted_sample_indices = sorted_indices
    
    # Extract sampled data
    y_proba_mc_sampled = y_proba_mc_full[sorted_sample_indices]
    y_test_sampled = y_test_full[sorted_sample_indices]
    conf_sampled = confident_mask[sorted_sample_indices]
    pred_lower_sampled = pred_risk_lower[sorted_sample_indices]
    pred_upper_sampled = pred_risk_upper[sorted_sample_indices]
    
    x_labels = [f"S{i}" for i in sorted_sample_indices]
    
    # Prepare custom boxplot data with CI whiskers
    boxplot_data = []
    positions = list(range(1, len(y_proba_mc_sampled) + 1))
    
    for i, sample_predictions in enumerate(y_proba_mc_sampled):
        # Traditional box statistics
        median = np.percentile(sample_predictions, 50)
        q1 = np.percentile(sample_predictions, 25)
        q3 = np.percentile(sample_predictions, 75)
        
        # Use CI bounds as whiskers instead of 1.5*IQR
        whisker_low = pred_lower_sampled[i]
        whisker_high = pred_upper_sampled[i]
        
        # Find outliers outside CI bounds
        outliers = sample_predictions[(sample_predictions < whisker_low) | 
                                    (sample_predictions > whisker_high)]
        
        boxplot_data.append({
            'med': median,
            'q1': q1,
            'q3': q3,
            'whislo': whisker_low,
            'whishi': whisker_high,
            'fliers': outliers
        })
    
    # Create the plot (same basic style as original)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create custom boxplot with CI whiskers
    bp = ax.bxp(boxplot_data, positions=positions, patch_artist=True,
                showfliers=True, widths=0.7,
                boxprops=dict(linewidth=1.2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                medianprops=dict(linewidth=2, color='white'),
                flierprops=dict(marker='o', markerfacecolor='gray', 
                               markersize=3, alpha=0.6, markeredgecolor='none'))
    
    # Color boxes based on confidence and true label (exact same as original)
    for i, (patch, true_label, is_confident) in enumerate(zip(bp['boxes'], y_test_sampled, conf_sampled)):
        if true_label == 1:  # Positive
            patch.set_facecolor('lightcoral' if is_confident else 'red')
        else:  # Negative
            patch.set_facecolor('lightgreen' if is_confident else 'darkgreen')
    
    # Add threshold line (same style as original)
    ax.axhline(y=prob_threshold, color='red', linestyle='--', linewidth=1.2)
    
    # Legend (exact same as original)
    legend_elements = [
        Patch(facecolor='lightcoral', label='Confident Positive'),
        Patch(facecolor='red', label='Inconfident Positive'),
        Patch(facecolor='lightgreen', label='Confident Negative'),
        Patch(facecolor='darkgreen', label='Inconfident Negative'),
        Patch(facecolor='white', edgecolor='red', label='Threshold (mean preds)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Title and labels (same style as original)
    ax.set_title(f"Age-split {task_name} - Sample Prediction Distributions (CI={ci_width})", fontsize=14)
    ax.set_ylabel("Prediction Probability", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # X-axis labels (same logic as original)
    if len(x_labels) > 20:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"Age-split {task_name}_ci{ci_width:.0%}_whisker_boxplot.png"
    plt.savefig(os.path.join(save_path, filename), dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ CI whisker boxplot saved: {filename}")

def create_ci_comparison_boxplot(task_name, 
                                 y_proba_mc_full, 
                                 y_test_full, 
                                 ci_widths=[0.75, 0.90],
                                 max_samples=100,
                                 save_path='results/boxplots_ci_comparison',
                                 figsize=(20, 8),
                                 dpi=300):
    """
    Create side-by-side comparison of different CI widths to visualize abstention changes.
    """
    
    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    
    # Calculate threshold as mean of all predictions
    # prob_threshold = y_proba_mc_full.mean(axis=1).mean()
    y_proba_mean = y_proba_mc_full.mean(axis=1)
    # mean_labels = np.mean(y_test_full)  # 使用标签平均值
    # prob_threshold = mean_labels
    prob_threshold = np.mean(y_proba_mean) 

    # Sort samples by mean probability
    n_samples = y_proba_mc_full.shape[0]
    sorted_indices = np.argsort(y_proba_mean)
    
    # Sample selection for display
    if n_samples > max_samples:
        sample_indices = np.linspace(0, n_samples - 1, max_samples, dtype=int)
        sorted_sample_indices = sorted_indices[sample_indices]
    else:
        sorted_sample_indices = sorted_indices
    
    # Extract sampled data
    y_proba_mc_sampled = y_proba_mc_full[sorted_sample_indices]
    y_test_sampled = y_test_full[sorted_sample_indices]
    x_labels = [f"S{i}" for i in sorted_sample_indices]
    
    # Create subplot figure
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.suptitle(f'Age-split Age-split {task_name} - CI Width Comparison: Abstention Analysis', 
                 fontsize=16, fontweight='bold')
    
    for idx, ci_width in enumerate(ci_widths):
        ax = axes[idx]
        
        # Calculate confidence intervals for this CI width
        lower_percentile = (1 - ci_width) / 2 * 100
        upper_percentile = (1 + ci_width) / 2 * 100
        pred_risk_lower = np.percentile(y_proba_mc_full[sorted_sample_indices], 
                                       lower_percentile, axis=1)
        pred_risk_upper = np.percentile(y_proba_mc_full[sorted_sample_indices], 
                                       upper_percentile, axis=1)
        
        # Determine confident predictions
        confident_positive = pred_risk_lower > prob_threshold
        confident_negative = pred_risk_upper < prob_threshold
        confident_mask = confident_positive | confident_negative
        
        # Prepare custom boxplot data with CI whiskers
        boxplot_data = []
        positions = list(range(1, len(y_proba_mc_sampled) + 1))
        
        for i, sample_predictions in enumerate(y_proba_mc_sampled):
            # Traditional box statistics
            median = np.percentile(sample_predictions, 50)
            q1 = np.percentile(sample_predictions, 25)
            q3 = np.percentile(sample_predictions, 75)
            
            # Use CI bounds as whiskers
            whisker_low = pred_risk_lower[i]
            whisker_high = pred_risk_upper[i]
            
            # Find outliers outside CI bounds
            outliers = sample_predictions[(sample_predictions < whisker_low) | 
                                        (sample_predictions > whisker_high)]
            
            boxplot_data.append({
                'med': median,
                'q1': q1,
                'q3': q3,
                'whislo': whisker_low,
                'whishi': whisker_high,
                'fliers': outliers
            })
        
        # Create boxplot
        bp = ax.bxp(boxplot_data, positions=positions, patch_artist=True,
                    showfliers=True, widths=0.7,
                    boxprops=dict(linewidth=1.2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    medianprops=dict(linewidth=2, color='white'),
                    flierprops=dict(marker='o', markerfacecolor='gray', 
                                   markersize=3, alpha=0.6, markeredgecolor='none'))
        
        # Color boxes based on confidence and true label (same as original)
        for i, (patch, true_label, is_confident) in enumerate(zip(bp['boxes'], y_test_sampled, confident_mask)):
            if true_label == 1:  # Positive
                patch.set_facecolor('lightcoral' if is_confident else 'red')
            else:  # Negative
                patch.set_facecolor('lightgreen' if is_confident else 'darkgreen')
        
        # Add threshold line
        ax.axhline(y=prob_threshold, color='red', linestyle='--', linewidth=1.2)
        
        # Title and labels for each subplot
        n_confident = np.sum(confident_mask)
        n_uncertain = len(confident_mask) - n_confident
        abstention_rate = n_uncertain / len(confident_mask) * 100
        
        ax.set_title(f'CI = {ci_width*100:.0f}%\n'
                    f'Confident: {n_confident}, Uncertain: {n_uncertain}\n'
                    f'Abstention Rate: {abstention_rate:.1f}%', 
                    fontsize=12)
        ax.set_ylabel("Prediction Probability" if idx == 0 else "", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # X-axis labels
        if len(x_labels) > 20:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        ax.set_xticks([])
        ax.set_xticklabels([])

        # Add legend only to the first subplot
        if idx == 0:
            legend_elements = [
                Patch(facecolor='lightcoral', label='Confident Positive'),
                Patch(facecolor='red', label='Inconfident Positive'),
                Patch(facecolor='lightgreen', label='Confident Negative'),
                Patch(facecolor='darkgreen', label='Inconfident Negative'),
                Patch(facecolor='white', edgecolor='red', label='Threshold (mean preds)')
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
            ax.set_xticks([])
            ax.set_xticklabels([])

    plt.tight_layout()
    
    # Save the comparison plot
    filename = f"Age-split {task_name}_ci_comparison_{'_'.join([str(int(cw*100)) for cw in ci_widths])}.png"
    plt.savefig(os.path.join(save_path, filename), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✓ CI comparison plot saved: {filename}")

def create_all_ci_comparisons(all_metrics, 
                             ci_widths=[0.75, 0.90],
                             save_path='results/boxplots_ci_comparison',
                             **kwargs):
    """
    Create CI comparison plots for all tasks.
    """
    
    print(f"Creating CI comparison plots ({ci_widths})...")
    
    for task_name, task_data in all_metrics.items():
        try:
            uncertainty_data = task_data['uncertainty']
            y_proba_mc_full = uncertainty_data['y_proba_mc_full']
            y_test_full = uncertainty_data['y_test_full']
            
            create_ci_comparison_boxplot(
                task_name=task_name,
                y_proba_mc_full=y_proba_mc_full,
                y_test_full=y_test_full,
                ci_widths=ci_widths,
                save_path=save_path,
                **kwargs
            )
            
        except Exception as e:
            print(f"Error creating CI comparison for Age-split {task_name}: {str(e)}")
    
    print(f"CI comparison plots saved to: {save_path}")

def analyze_uncertainty_remove_rates(task_name, y_test, y_proba_mean, epistemic_uncertainty,
                                     remove_rates=[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
    """Analyze performance after removing uncertain samples by different rates"""
    prob_threshold = np.mean(y_proba_mean)
    y_pred = (y_proba_mean > prob_threshold).astype(int)

    results = []

    for remove_rate in remove_rates:
        if remove_rate == 0:
            # Keep all samples
            keep_mask = np.ones(len(y_test), dtype=bool)
        else:
            # Remove top remove_rate% most uncertain samples
            # Sort uncertainty and keep the lowest (1-remove_rate) portion
            n_keep = int((1 - remove_rate) * len(epistemic_uncertainty))
            keep_indices = np.argsort(epistemic_uncertainty)[:n_keep]
            keep_mask = np.zeros(len(y_test), dtype=bool)
            keep_mask[keep_indices] = True

        # Get remaining samples
        y_test_keep = y_test[keep_mask]
        y_pred_keep = y_pred[keep_mask]
        y_proba_keep = y_proba_mean[keep_mask]

        # Calculate metrics
        if len(y_test_keep) == 0 or len(np.unique(y_test_keep)) < 2:
            tpr = tnr = fpr = fnr = np.nan
            n_remaining = 0
        else:
            cm = confusion_matrix(y_test_keep, y_pred_keep, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                # Handle case where only one class remains
                if np.all(y_test_keep == 0):  # Only negatives
                    tn = np.sum(y_pred_keep == 0)
                    fp = np.sum(y_pred_keep == 1)
                    fn = tp = 0
                else:  # Only positives
                    tp = np.sum(y_pred_keep == 1)
                    fn = np.sum(y_pred_keep == 0)
                    tn = fp = 0

            tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            fpr = fp / (tn + fp) if (tn + fp) > 0 else np.nan
            fnr = fn / (tp + fn) if (tp + fn) > 0 else np.nan
            n_remaining = len(y_test_keep)

        abstention_rate = 1 - (len(y_test_keep) / len(y_test))

        results.append({
            'task_name': task_name,
            'remove_rate': remove_rate,
            'abstention_rate': abstention_rate,
            'n_remaining': n_remaining,
            'tpr': tpr,
            'tnr': tnr,
            'fpr': fpr,
            'fnr': fnr,
            'method': 'entropy'
        })

    return results

def main():
    """
    Comprehensive main function implementing all required analyses and visualizations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    base_dir = "/hpc/group/engelhardlab/xg97"
    # base_dir = "D:/DUKE/LAB/Matt/femr_cuda_env"
    tasks_path = f"{base_dir}/EHRSHOT_ASSETS/benchmark"
    features_path = f"{base_dir}/EHRSHOT_ASSETS/features/clmbr_features.pkl"
    splits_path = f"{base_dir}/EHRSHOT_ASSETS/splits/person_id_map.csv"
    age_path = f"{base_dir}/EHRSHOT_ASSETS/patient_age_from_metadata.csv"

    base_save_dir = f"{base_dir}/results_age_0901"
    os.makedirs(base_save_dir, exist_ok=True)

    os.makedirs(f"{base_save_dir}/ci_analysis", exist_ok=True)
    os.makedirs(f"{base_save_dir}/entropy_analysis", exist_ok=True)
    os.makedirs(f"{base_save_dir}/boxplots_single", exist_ok=True)
    os.makedirs(f"{base_save_dir}/boxplots_comparison", exist_ok=True)
    
    if not os.path.exists(tasks_path):
        print(f"ERROR: Cannot find tasks directory at: {tasks_path}")
        return
    
    task_dirs = [d for d in os.listdir(tasks_path) if os.path.isdir(os.path.join(tasks_path, d))]
    print(f"Found {len(task_dirs)} tasks: {task_dirs}")
    
    # Load features
    print("Loading features...")
    features_df, key_global = load_features()
    if features_df is None or key_global is None:
        print("ERROR: Failed to load features")
        return
    
    # Process all tasks
    print("Processing tasks...")
    all_metrics = {}
    failed_tasks = []
    
    for task_name in task_dirs:
        print(f"\nProcessing task: Age-split {task_name}")
        try:
            metrics_by_e_v, unc_dict = process_task(
                task_name, features_df, key_global, device, 
                e_v_thresholds=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
            )
            all_metrics[task_name] = {
                'metrics_by_e_v': metrics_by_e_v,
                'uncertainty': unc_dict
            }
            print(f"  ✓ Successfully processed Age-split {task_name}")
        except Exception as e:
            print(f"  ✗ Error processing Age-split {task_name}: {str(e)}")
            failed_tasks.append(task_name)
    
    if len(all_metrics) == 0:
        print("ERROR: No tasks were successfully processed. Exiting.")
        return
    
    print(f"\nSuccessfully processed {len(all_metrics)} tasks")
    if failed_tasks:
        print(f"Failed tasks: {failed_tasks}")
    
    # ==================== ANALYSIS 1: CI-based Analysis ====================
    print("\n" + "="*60)
    print("1. CI-BASED UNCERTAINTY ANALYSIS")
    print("="*60)
    
    ci_analysis_results = {}
    ci_widths = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    for task_name, task_data in all_metrics.items():
        print(f"Analyzing CI effects for Age-split {task_name}...")
        try:
            uncertainty_data = task_data['uncertainty']
            y_test = uncertainty_data['y_test_full']
            y_proba_mc = uncertainty_data['y_proba_mc_full']
            
            # Create predictions_dict format for analysis
            predictions_dict = {
                'y_test': y_test,
                'y_proba_mc': y_proba_mc
            }
            
            # Analyze CI width effects
            ci_results = analyze_ci_width_effects(task_name, predictions_dict)
            ci_analysis_results[task_name] = ci_results
            
        except Exception as e:
            print(f"  Error in CI analysis for Age-split {task_name}: {str(e)}")

    # Save CI analysis results to Excel
    print("Saving CI analysis results...")
    try:
        save_improved_results_to_excel(ci_analysis_results, f"{base_save_dir}/ci_analysis")
        print("  ✓ CI analysis Excel saved")
    except Exception as e:
        print(f"  ✗ Error saving CI Excel: {str(e)}")

    
    # Create CI analysis plots
    print("Creating CI analysis plots...")
    for task_name, task_data in ci_analysis_results.items():
        try:
            create_improved_ci_width_plots(task_name, task_data, f"{base_save_dir}/ci_analysis")
        except Exception as e:
            print(f"  Error creating CI plot for Age-split {task_name}: {str(e)}")
    
    # ==================== ANALYSIS 2: Entropy-based Analysis ====================
    entropy_remove_results = []
    for task_name, task_data in all_metrics.items():
        uncertainty_data = task_data['uncertainty']
        y_test = uncertainty_data['y_test_full']
        y_proba_mc = uncertainty_data['y_proba_mc_full']
        y_proba_mean = y_proba_mc.mean(axis=1)
        epistemic_uncertainty = uncertainty_data['test_epistemic_uncertainty']

    # Use the analyze_uncertainty_remove_rates function
        task_results = analyze_uncertainty_remove_rates(
            task_name=task_name,
            y_test=y_test,
            y_proba_mean=y_proba_mean,
            epistemic_uncertainty=epistemic_uncertainty,
            remove_rates=[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        )

        entropy_remove_results.extend(task_results)

    df_remove_metrics = pd.DataFrame(entropy_remove_results)
    df_remove_metrics.to_csv(f"{base_save_dir}/entropy_analysis/Age_split_entropy_remove_rates_analysis.csv", index=False)
    print(f"Saved entropy remove rates analysis → {base_save_dir}/entropy_analysis/Age_split_entropy_remove_rates_analysis.csv")
    # Use e_v_thresholds to analyze abstention rates
    entropy_analysis_results = []
    for task_name, task_data in all_metrics.items():
        metrics_list = task_data['metrics_by_e_v']
        for m in metrics_list:
            entropy_analysis_result = {'task_name': task_name}
            entropy_analysis_result.update(m)
            entropy_analysis_results.append(entropy_analysis_result)

    df_metrics = pd.DataFrame(entropy_analysis_results)
    drop_cols = ['y_test_keep', 'y_proba_mean_keep', 'y_proba_mc',
             'y_pred', 'epistemic_uncertainty',
             'aleatoric_entropy', 'predictive_entropy']
    df_metrics = df_metrics.drop(columns=[c for c in drop_cols if c in df_metrics.columns])
    df_metrics.to_csv(f"{base_save_dir}/entropy_analysis/Age_split_main_uncertainty_metrics_summary.csv", index=False)
    print(f"Saved metrics_by_e_v → {base_save_dir}/entropy_analysis/Age_split_main_uncertainty_metrics_summary.csv")

    # ==================== ANALYSIS 3: Individual Boxplots ====================
    print("\n" + "="*60)
    print("3. INDIVIDUAL TASK BOXPLOTS")
    print("="*60)
    
    # Single CI whisker boxplots (CI=0.95)
    print("Creating individual CI whisker boxplots (CI=95%)...")
    for task_name, task_data in all_metrics.items():
        try:
            uncertainty_data = task_data['uncertainty']
            y_proba_mc_full = uncertainty_data['y_proba_mc_full']
            y_test_full = uncertainty_data['y_test_full']
            
            create_ci_whisker_boxplot(
                task_name=task_name,
                y_proba_mc_full=y_proba_mc_full,
                y_test_full=y_test_full,
                ci_width=0.95,
                save_path=f"{base_save_dir}/boxplots_single",
                max_samples=100,
                figsize=(15, 8)
            )
        except Exception as e:
            print(f"  Error creating boxplot for Age-split {task_name}: {str(e)}")
    
    # ==================== ANALYSIS 4: Comparison Boxplots ====================
    print("\n" + "="*60)
    print("4. CI COMPARISON BOXPLOTS")
    print("="*60)
    
    # Create comparison boxplots (75% vs 90%)
    print("Creating CI comparison boxplots (75% vs 90%)...")
    create_all_ci_comparisons(
        all_metrics,
        ci_widths=[0.75, 0.90],
        save_path=f"{base_save_dir}/boxplots_comparison",
        max_samples=100,
        figsize=(20, 8)
    )
    
    # Also create 90% vs 95% comparison
    print("Creating CI comparison boxplots (90% vs 95%)...")
    create_all_ci_comparisons(
        all_metrics,
        ci_widths=[0.90, 0.95],
        save_path=f"{base_save_dir}/boxplots_comparison",
        max_samples=100,
        figsize=(20, 8)
    )

    
   
    # ==================== FINAL SUMMARY ====================
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Results saved to: {base_save_dir}")
    print("\nGenerated outputs:")
    print("📊 CI Analysis:")
    print(f"  - {base_save_dir}/ci_analysis/ci_width_analysis_results.xlsx")
    print(f"  - {base_save_dir}/ci_analysis/plots/*_improved_ci_analysis.png")
    print("\n📈 Entropy Analysis:")
    # print(f"  - {base_save_dir}/entropy_analysis/entropy_abstention_analysis.csv")
    # print(f"  - {base_save_dir}/entropy_analysis/entropy_remove_rates.csv")
    # print(f"  - {base_save_dir}/entropy_analysis/ci_abstention_rates.csv")
    print("\n📦 Individual Boxplots:")
    print(f"  - {base_save_dir}/boxplots_single/*_ci95%_whisker_boxplot.png")
    print("\n🔄 Comparison Boxplots:")
    print(f"  - {base_save_dir}/boxplots_comparison/*_ci_comparison_75_90.png")
    print(f"  - {base_save_dir}/boxplots_comparison/*_ci_comparison_90_95.png")
    print("\n📋 Summary Statistics:")
    print(f"  - {base_save_dir}/main_metrics_summary.csv")

    
    print(f"\nProcessed {len(all_metrics)} tasks successfully.")
    if failed_tasks:
        print(f"Failed tasks: {failed_tasks}")

if __name__ == "__main__":
    main()
