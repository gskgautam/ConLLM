import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from models.classification import ClassifierHead, build_label_map, RefinedEmbeddingDataset

def compute_eer(labels, scores):
    # Compute Equal Error Rate (EER)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    idx_eer = np.nanargmin(abs_diffs)
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2
    return eer

def evaluate_classifier(embedding_dir, model_path, metric='acc', batch_size=64, device='cuda'):
    label_map = build_label_map(embedding_dir)
    dataset = RefinedEmbeddingDataset(embedding_dir, label_map)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # Infer input_dim from a sample embedding
    sample_file = next((f for f in os.listdir(embedding_dir) if f.endswith('.npy')), None)
    if sample_file is None:
        print('No embeddings found.')
        return
    sample_emb = np.load(os.path.join(embedding_dir, sample_file))
    input_dim = sample_emb.shape[-1]
    model = ClassifierHead(input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for emb, label in dataloader:
            emb = emb.to(device)
            pred = model(emb).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(label.numpy())
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    if metric == 'eer':
        eer = compute_eer(all_labels, all_preds)
        print(f"[RESULT] EER: {eer:.4f}")
        return eer
    else:
        acc = accuracy_score(all_labels, np.round(all_preds))
        auc = roc_auc_score(all_labels, all_preds)
        print(f"[RESULT] Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        return acc, auc

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Example: evaluate audio (EER)
    audio_dir = '../embeddings/audio_refined/'
    audio_model = os.path.join(audio_dir, 'classifier_head.pt')
    evaluate_classifier(audio_dir, audio_model, metric='eer', device=device)
    # Example: evaluate video (ACC, AUC)
    video_dir = '../embeddings/video_refined/'
    video_model = os.path.join(video_dir, 'classifier_head.pt')
    evaluate_classifier(video_dir, video_model, metric='acc', device=device)
    # Example: evaluate AV (ACC, AUC)
    av_dir = '../embeddings/av_refined/'
    av_model = os.path.join(av_dir, 'classifier_head.pt')
    evaluate_classifier(av_dir, av_model, metric='acc', device=device) 