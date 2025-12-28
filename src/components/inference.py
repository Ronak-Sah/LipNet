import os
from src.logger import logger
from src.entity import ModelEvaluationConfig
import torch
import torch.nn as nn
from torch.utils.data import  Dataset,DataLoader
from src.components.model.transformer import Transformer
from src.components.preprocessing import Tokenizer,Loader,Frames



tokenizer=Tokenizer()
vocab_len=len(tokenizer.idx_to_char)


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

def chunk_frames(frames, chunk_size=200, overlap=50):
    chunks = []
    i = 0
    while i + chunk_size <= frames.shape[0]:
        chunks.append(frames[i:i+chunk_size])
        i += chunk_size - overlap
    return chunks



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

        self.model.load_state_dict(checkpoint["model_state"], strict=False)
        # self.max_length=self.config.max_length

    def predict(self, video_path):
        self.model.eval()
        # self.max_length=self.config.max_length
        load_frames=Frames(self.config.landmark_model_path)
        frames=load_frames.extract_mouth_frames(video_path)
        print("Frames ",frames.shape)
        if frames is None:
            return ""
        
        all_preds_tokens = []
        if frames.shape[0] < 200:
            chunks = torch.tensor(frames, dtype=torch.float32)  
        else:
            chunks = chunk_frames(frames, 200, overlap=50)
            chunks = torch.tensor(chunks[0], dtype=torch.float32)
        # print("chunks ",chunks[0].shape)
        chunks = torch.tensor(chunks, dtype=torch.float32)
        chunks = chunks.unsqueeze(0)
        print("chunks ",chunks.shape)
        pred_text = ""

        with torch.no_grad():
            for chunk in chunks:
                chunk = chunk.unsqueeze(0).unsqueeze(0).to(self.device)
                print("chunks ",chunks.shape)
                # X = frames.unsqueeze(0).to(self.device)
                # if isinstance(X, torch.Tensor):
                #     X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-8)
                y_pred=self.model(chunk)

                decoded_tokens = ctc_greedy_decode(y_pred)

                all_preds_tokens.extend(decoded_tokens[0])



        pred_text += tokenizer.labels_to_text(all_preds_tokens)

        return pred_text