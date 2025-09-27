# 🧠 Emotion Recognition in Python


![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

This project implements **emotion recognition on tweets** using the Hugging Face `emotion` dataset.  
It combines **Transformer-based tokenization** with a **custom BiLSTM architecture in PyTorch** to classify text into 6 emotion categories.

---

## 📌 Features
- Hugging Face `datasets` integration (`emotion` dataset).
- Preprocessing with `AutoTokenizer` (`bert-base-uncased`).
- Custom **BiLSTM classifier** implemented in PyTorch.
- Training loop with **early stopping**.
- Visualizations: 
  - Confusion matrix
  - Training vs. validation performance

---

## 🏗️ Architectures

<details>
<summary>🔹 BiLSTM Classifier</summary>

```python
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=20, num_classes=6):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, (h, _) = self.lstm2(x)
        h_final = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(h_final)
```
</details>

# ⚙️ Setup
## Clone repo

```git
git clone <this-repo>
cd modernised-emotion-recognition
```

# Install dependencies
```python
pip install torch torchvision torchaudio
pip install transformers datasets matplotlib scikit-learn
```

# 🚀 Training
## Train with early stopping
```python
epochs = 20
patience = 3
```

Training stops if validation accuracy does not improve for 3 consecutive epochs.

## 📊 Results & Evaluation

Training and validation curves plotted over epochs.

Confusion matrix for per-class performance.

Final evaluation on test set using accuracy and loss.

# 🛠️ Future Work

Experiment with Transformer-based fine-tuning (BERT, RoBERTa).
Hyperparameter tuning for embeddings & LSTM hidden size.
Deploy model via FastAPI or Gradio for real-time demos.