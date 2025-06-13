import torch
from torch import nn
import numpy as np

class PolyTransformer(nn.Module):
    def __init__(self, n_tracks=4, d_model=128, num_lstm_layers=2, dropout_rate=0.2):
        super().__init__()
        self.note_embed = nn.Embedding(128, d_model // 2)
        self.track_embed = nn.Embedding(n_tracks, d_model // 4)
        self.continuous_linear = nn.Linear(3, d_model // 4)
        lstm_input_size = (d_model // 2) + (d_model // 4) + (d_model // 4)
        self.lstm = nn.LSTM(lstm_input_size, d_model, num_layers=num_lstm_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(d_model, n_tracks * 5)

    def forward(self, x):
        note = self.note_embed(x[:, :, 1].long())
        track = self.track_embed(x[:, :, 4].long())
        continuous = self.continuous_linear(x[:, :, :3].float())
        x = torch.cat([note, track, continuous], dim=-1)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.output(out)
        return out


def train_poly_model(dataset, params):
    import streamlit as st
    try:
        n_tracks = params.get("n_tracks", 4)
        model = PolyTransformer(n_tracks=n_tracks)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get("lr", 0.001))
        loss_fn = nn.MSELoss()
        loader = torch.utils.data.DataLoader(dataset, batch_size=params.get("batch_size", 1), shuffle=True)
        for epoch in range(params.get("epochs", 10)):
            total_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                out = model(batch)
                loss = loss_fn(out, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss
            if (epoch + 1) % 10 == 0:
                st.write(f"Epoch {epoch+1}/{params['epochs']}, Total Loss: {total_loss.item():.4f}")
        torch.save(model.state_dict(), params["model_path"])
        return model
    except RuntimeError as e:
        st.error(f"Erro ao treinar o modelo de IA: {e}")
        return None


def infer_poly_sequence(model, seq_len=512, n_tracks=4, prime_sequence=None, randomness_temp=1.0):
    import streamlit as st
    try:
        model.eval()
        if prime_sequence is None:
            prime_sequence = torch.zeros((1, 1, 5), dtype=torch.float)
        generated = [prime_sequence]
        for _ in range(seq_len):
            inp = torch.cat(generated, dim=1)
            out = model(inp)
            next_step = out[:, -1, :].view(1, n_tracks, 5)
            generated.append(next_step)
        seq = torch.cat(generated, dim=1).squeeze(0)
        out_sequences_by_track = {i: [] for i in range(n_tracks)}
        cumulative = torch.zeros(n_tracks)
        for step in seq:
            delta, note, vel, dur, track = step
            track = int(track.item())
            cumulative[track] += delta.item()
            out_sequences_by_track[track].append([delta.item(), note.item(), vel.item(), dur.item(), track])
        return [np.array(out_sequences_by_track[k]) for k in sorted(out_sequences_by_track.keys())]
    except RuntimeError as e:
        st.error(f"Erro ao inferir sequência com o modelo de IA: {e}")
        return []
