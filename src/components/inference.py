import os
from src.logger import logger
from src.entity import ModelEvaluationConfig
import torch
import torch.nn as nn
from torch.utils.data import  Dataset,DataLoader
from src.components.model.transformer import Transformer
from src.components.preprocessing import Tokenizer,Loader,Frames



tokenizer=Tokenizer()
vocab_len=len(tokenizer.vocab)


import torch

def ctc_greedy_decode(logits, blank=0):

    preds = torch.argmax(logits, dim=-1)  # (B, T)

    decoded = []
    for seq in preds:
        prev = blank
        out = []
        for token in seq:
            token = token.item()
            if token != prev and token != blank:
                out.append(token)
            prev = token
        decoded.append(out)

    return decoded




class ModelPrediction:
    def __init__(self,config: ModelEvaluationConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Transformer(
            ffn_hidden=self.config.ffn_hidden,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            max_length=self.config.max_length,
            vocab_size=vocab_len
        ).to(self.device)

        model_path = config.model_path
        # self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        checkpoint_path = os.path.join("artifacts\model_trainer\checkpoint.pth")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state"])

    def predict(self, video_path):
        self.model.eval()
        
        load_frames=Frames(self.config.landmark_model_path)
        frames=load_frames.extract_mouth_frames(video_path)
        if frames is None:
            return ""
        frames = frames.unsqueeze(0)
        frames = torch.tensor(frames, dtype=torch.float32)

        with torch.no_grad():
            X = frames.unsqueeze(0).to(self.device)

            y_pred=self.model(X)

            decoded_tokens = ctc_greedy_decode(y_pred)

            pred_text = tokenizer.labels_to_text(decoded_tokens[0])

        return pred_text