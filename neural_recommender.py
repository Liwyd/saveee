import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


class EmbeddingModel(nn.Module):
    def __init__(self, input_dim, embed_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


class NeuralRecommender:
    def __init__(self, csv_path, embed_dim=32, epochs=100, batch_size=16, lr=1e-3):
        self.df = pd.read_csv(csv_path)
        self.song_names = self.df["file"].values
        self.features = self.df.drop(columns=["file"]).values.astype(np.float32)

        # Scale the features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

        # Convert to torch tensor
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.loader = torch.utils.data.DataLoader(self.features, batch_size=batch_size, shuffle=True)

        input_dim = self.features.shape[1]
        self.model = EmbeddingModel(input_dim, embed_dim)
        self.embeddings = None
        self.train(epochs, lr)

    def train(self, epochs, lr):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for batch in self.loader:
                optimizer.zero_grad()
                z, x_hat = self.model(batch)
                loss = loss_fn(x_hat, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

        # Store the embeddings
        with torch.no_grad():
            self.embeddings, _ = self.model(self.features)

    def recommend(self, song_name, top_k=5):
        if self.embeddings is None:
            raise ValueError("Model not trained or embeddings not generated.")

        index = np.where(self.song_names == song_name)[0]
        if len(index) == 0:
            raise ValueError(f"Song '{song_name}' not found in dataset.")

        index = index[0]
        query_embedding = self.embeddings[index].unsqueeze(0)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = similarities.argsort()[::-1][1:top_k+1]
        return [self.song_names[i] for i in top_indices]