import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader
import numpy as np
import lmdb
import pickle
import os
from tqdm import tqdm


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length = int(txn.get(b"__len__").decode())
            self.dim = int(txn.get(b"__dim__").decode())
            files_pickle = txn.get(b"__files__")  # اسم ترک‌ها به صورت pickle ذخیره شدن
            if files_pickle is not None:
                self.files = pickle.loads(files_pickle)
            else:
                self.files = []

    def __getitem__(self, index):
        with self.env.begin() as txn:
            key = f"{index}".encode()
            byteflow = txn.get(key)
            sample = np.frombuffer(byteflow, dtype=np.float32)
            return torch.tensor(sample)

    def __len__(self):
        return self.length



class EmbeddingModel(nn.Module):
    def __init__(self, input_dim, embed_dim=32, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


class NeuralRecommender:
    def __init__(self, lmdb_path, model_path="recommender.pt", embed_dim=32,
                 epochs=10, batch_size=256, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = LMDBDataset(lmdb_path)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.input_dim = self.dataset.dim
        self.model = EmbeddingModel(self.input_dim, embed_dim).to(self.device)
        self.embeddings = None
        self.model_path = model_path

        # ساخت دیکشنری اسم ترک -> ایندکس
        self.file_to_index = {fname: idx for idx, fname in enumerate(self.dataset.files)}

        if os.path.exists(model_path):
            self.load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
        else:
            self.train(epochs, lr)
            self.save_model(model_path)

    def train(self, epochs, lr):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(self.loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch = batch.to(self.device)
                optimizer.zero_grad()
                z, x_hat = self.model(batch)
                loss = loss_fn(x_hat, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Loss: {total_loss:.4f}")

        self.build_embeddings()

    def build_embeddings(self):
        self.model.eval()
        all_embeddings = []
        loader = DataLoader(self.dataset, batch_size=1024, shuffle=False, num_workers=4)

        with torch.no_grad():
            for batch in tqdm(loader, desc="Building embeddings"):
                batch = batch.to(self.device)
                z, _ = self.model(batch)
                all_embeddings.append(z.cpu())

        self.embeddings = torch.cat(all_embeddings, dim=0)


    def recommend(self, query, top_k=5):
        if self.embeddings is None:
            raise ValueError("Embeddings not generated.")

        # ورودی اسم ترک باشه
        if isinstance(query, str):
            if query not in self.file_to_index:
                raise ValueError(f"Track name '{query}' not found in dataset.")
            index = self.file_to_index[query]
            query_embedding = self.embeddings[index].unsqueeze(0)
        else:
            raise TypeError("Query must be a track name string.")

        sims = cosine_similarity(query_embedding.cpu(), self.embeddings.cpu())[0]
        top_indices = sims.argsort()[::-1][1:top_k+1]

        # برگردوندن لیست (ایندکس، اسم ترک)
        return [(idx, self.dataset.files[idx]) for idx in top_indices]
    
    
    def save_model(self, path):
        torch.save({
            'model_state': self.model.state_dict(),
            'embeddings': self.embeddings.cpu()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        self.embeddings = checkpoint['embeddings']


# Example usage:
if __name__ == "__main__":
    recommender = NeuralRecommender("music_embeddings.lmdb", epochs=5)
    print("Top similar items to index 42:", recommender.recommend(42))
