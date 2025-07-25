import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve
from models.classification import ClassifierHead, build_label_map, RefinedEmbeddingDataset

def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    idx_eer = np.nanargmin(abs_diffs)
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2
    return eer

def train_on_language(train_dir, language, device='cuda'):
    # Train classifier on one language
    label_map = build_label_map(train_dir)
    dataset = RefinedEmbeddingDataset(train_dir, label_map)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    sample_file = next((f for f in os.listdir(train_dir) if f.endswith('.npy')), None)
    sample_emb = np.load(os.path.join(train_dir, sample_file))
    input_dim = sample_emb.shape[-1]
    model = ClassifierHead(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()
    model.train()
    for epoch in range(5):
        for emb, label in dataloader:
            emb = emb.to(device)
            label = label.to(device)
            pred = model(emb)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # Save model
    save_path = os.path.join(train_dir, f'classifier_{language}.pt')
    torch.save(model.state_dict(), save_path)
    return model, input_dim

def test_on_language(model, input_dim, test_dir, device='cuda'):
    # Test classifier on another language
    label_map = build_label_map(test_dir)
    dataset = RefinedEmbeddingDataset(test_dir, label_map)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    model = model.to(device)
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for emb, label in dataloader:
            emb = emb.to(device)
            pred = model(emb).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(label.numpy())
    eer = compute_eer(np.array(all_labels), np.array(all_preds))
    return eer

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Directories for DECRO English and Chinese refined embeddings
    en_dir = '../embeddings/decro_English_refined/'
    zh_dir = '../embeddings/decro_Chinese_refined/'
    # Train on English, test on Chinese
    print('Cross-lingual: Train on English, Test on Chinese')
    model_en, input_dim_en = train_on_language(en_dir, 'English', device=device)
    eer_en2zh = test_on_language(model_en, input_dim_en, zh_dir, device=device)
    print(f'EER (EN->ZH): {eer_en2zh:.4f}')
    # Train on Chinese, test on English
    print('Cross-lingual: Train on Chinese, Test on English')
    model_zh, input_dim_zh = train_on_language(zh_dir, 'Chinese', device=device)
    eer_zh2en = test_on_language(model_zh, input_dim_zh, en_dir, device=device)
    print(f'EER (ZH->EN): {eer_zh2en:.4f}') 