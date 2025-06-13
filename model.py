import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from collections import defaultdict

class PolyTransformer(nn.Module):
    def __init__(self, n_tracks=4, d_model=128, num_lstm_layers=2, dropout_rate=0.2):
        super().__init__()
        self.note_embed = nn.Embedding(128, d_model // 2)
        self.track_embed = nn.Embedding(n_tracks, d_model // 4)
        self.continuous_linear = nn.Linear(3, d_model // 4)
        lstm_input_size = (d_model // 2) + (d_model // 4) + (d_model // 4)
        self.lstm = nn.LSTM(lstm_input_size, d_model, num_layers=num_lstm_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_delta_tick = nn.Linear(d_model, 1)
        self.output_note = nn.Linear(d_model, 128)
        self.output_velocity = nn.Linear(d_model, 1)
        self.output_duration = nn.Linear(d_model, 1)
        self.output_track = nn.Linear(d_model, n_tracks)

    def forward(self, x):
        delta_tick_data = x[:, :, 0].unsqueeze(-1)
        note_data = x[:, :, 1].long()
        velocity_data = x[:, :, 2].unsqueeze(-1)
        duration_data = x[:, :, 3].unsqueeze(-1)
        track_data = x[:, :, 4].long()
        note_embedded = self.note_embed(note_data)
        track_embedded = self.track_embed(track_data)
        continuous_features = torch.cat([delta_tick_data, velocity_data, duration_data], dim=-1)
        continuous_projected = self.continuous_linear(continuous_features.float())
        combined_input = torch.cat([note_embedded, track_embedded, continuous_projected], dim=-1)
        lstm_out, _ = self.lstm(combined_input)
        lstm_out = self.dropout(lstm_out)
        pred_delta_tick = F.relu(self.output_delta_tick(lstm_out)) + 1
        pred_note_logits = self.output_note(lstm_out)
        pred_velocity = torch.sigmoid(self.output_velocity(lstm_out)) * 127
        pred_duration = F.relu(self.output_duration(lstm_out)) + 1
        pred_track_logits = self.output_track(lstm_out)
        return pred_delta_tick, pred_note_logits, pred_velocity, pred_duration, pred_track_logits


def train_poly_model(notes_data, params):
    try:
        if notes_data.shape[0] == 0:
            st.warning("Nenhuma nota para treinar o modelo de IA.")
            return None
        data_tensor = torch.from_numpy(notes_data.astype(float)).float().unsqueeze(0)
        n_tracks_in_data = int(notes_data[:, 4].max()) + 1 if notes_data.shape[0] > 0 else 1
        model = PolyTransformer(
            n_tracks=n_tracks_in_data,
            d_model=params.get("d_model", 128),
            num_lstm_layers=params.get("num_lstm_layers", 2),
            dropout_rate=params.get("dropout_rate", 0.2),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        criterion_continuous = nn.MSELoss()
        criterion_note = nn.CrossEntropyLoss()
        criterion_track = nn.CrossEntropyLoss()
        for epoch in range(params["epochs"]):
            optimizer.zero_grad()
            pred_delta_tick, pred_note_logits, pred_velocity, pred_duration, pred_track_logits = model(data_tensor)
            target_delta_tick = data_tensor[:, :, 0].unsqueeze(-1)
            target_note = data_tensor[:, :, 1].long()
            target_velocity = data_tensor[:, :, 2].unsqueeze(-1)
            target_duration = data_tensor[:, :, 3].unsqueeze(-1)
            target_track = data_tensor[:, :, 4].long()
            loss_delta_tick = criterion_continuous(pred_delta_tick, target_delta_tick)
            loss_note = criterion_note(pred_note_logits.view(-1, 128), target_note.view(-1))
            loss_velocity = criterion_continuous(pred_velocity, target_velocity)
            loss_duration = criterion_continuous(pred_duration, target_duration)
            loss_track = criterion_track(pred_track_logits.view(-1, n_tracks_in_data), target_track.view(-1))
            total_loss = loss_delta_tick + loss_note + loss_velocity + loss_duration + loss_track
            total_loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                st.write(f"Epoch {epoch+1}/{params['epochs']}, Total Loss: {total_loss.item():.4f}")
        torch.save(model.state_dict(), params["model_path"])
        return model
    except Exception as e:
        st.error(f"Erro ao treinar o modelo de IA: {e}")
        return None


def infer_poly_sequence(model, seq_len=512, n_tracks=4, prime_sequence=None, randomness_temp=1.0):
    try:
        model.eval()
        if prime_sequence is None or prime_sequence.shape[0] == 0:
            prime_sequence = np.array([[100, 60, 80, 100, 0]], dtype=float)
        current_sequence = prime_sequence.copy()
        for _ in range(seq_len):
            input_note = torch.from_numpy(current_sequence[-1:].astype(float)).float().unsqueeze(0)
            with torch.no_grad():
                pred_delta_tick, pred_note_logits, pred_velocity, pred_duration, pred_track_logits = model(input_note)
            pred_delta_tick_val = max(1, int(pred_delta_tick.item()))
            pred_note_prob = F.softmax(pred_note_logits / randomness_temp, dim=-1)
            pred_note_val = torch.multinomial(pred_note_prob.squeeze(0), 1).item()
            pred_velocity_val = int(np.clip(pred_velocity.item(), 0, 127))
            pred_duration_val = max(1, int(pred_duration.item()))
            pred_track_prob = F.softmax(pred_track_logits / randomness_temp, dim=-1)
            pred_track_val = torch.multinomial(pred_track_prob.squeeze(0), 1).item()
            new_note_features = np.array([pred_delta_tick_val, pred_note_val, pred_velocity_val, pred_duration_val, pred_track_val], dtype=float)
            current_sequence = np.vstack([current_sequence, new_note_features])
        out_sequences_by_track = defaultdict(list)
        for note_event in current_sequence:
            out_sequences_by_track[int(note_event[4])].append(note_event)
        final_out_sequences = [np.array(out_sequences_by_track[k]) for k in sorted(out_sequences_by_track.keys())]
        return final_out_sequences
    except Exception as e:
        st.error(f"Erro ao inferir sequência com o modelo de IA: {e}")
        return []
