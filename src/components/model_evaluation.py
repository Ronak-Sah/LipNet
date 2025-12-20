import os
from src.logger import logger
from src.entity import ModelEvaluationConfig
import torch
import torch.nn as nn
from torch.utils.data import  Dataset,DataLoader
from src.components.models.transformer import Transformer
from src.components.preprocessing import Tokenizer,Loader



class Cnn_Dataset(Dataset):
    def __init__(self, X_path, y_path, limit=100):
        self.X_path = X_path[1000:1000+limit]
        self.y_path = y_path[1000:1000+limit]

    def __len__(self):
        return len(self.X_path)

    def __getitem__(self, idx):
        return self.X_path[idx], self.X_path[idx].split("\\")[-1][0:-5]+"mpg"


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


def word_error_rate(ref, hyp):
    """
    ref: ground truth string
    hyp: predicted string
    """
    ref_words = ref.split()
    hyp_words = hyp.split()

    n = len(ref_words)
    m = len(hyp_words)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,    # deletion
                    dp[i][j - 1] + 1,    # insertion
                    dp[i - 1][j - 1] + 1 # substitution
                )

    return dp[n][m] / max(1, n)



class Model_Evaluation:
    def __init__(self,config: ModelEvaluationConfig,):
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
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))


        logger.info("Model loaded succesfully for evaluation")

    
    def evaluate(self):
        
        self.model.eval()
        total_wer = 0.0
        count = 0
        batch_no=0

        alignment_path=os.path.join(self.config.alignment_data_path)
        speaker_path=os.path.join(self.config.speaker_data_path)

        X_paths=os.listdir(alignment_path)
        y_paths=os.listdir(speaker_path)

        loader=Loader()

        dataset = Cnn_Dataset(X_paths, y_paths)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,  
            shuffle=True
        )

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                batch_no=batch_no+1
                rem=(100//self.config.batch_size)-batch_no+1
                print(f"Batch no : {batch_no}, Total batch : {100/self.config.batch_size}, Remaining batch :{rem}" )
                X,y=loader.load_data(X_batch, y_batch, self.config.alignment_data_path,self.config.speaker_data_path,self.config.landmark_model_path)
                
                X = X.to(self.device)
                y = [t.to(self.device) for t in y]


                y_pred=self.model(X)
                # y_pred = y_pred.permute(1, 0, 2)
 

                decoded_tokens = ctc_greedy_decode(y_pred)

                for i in range(len(decoded_tokens)):
                    pred_text = tokenizer.labels_to_text(decoded_tokens[i])
                    
                    gt_text = tokenizer.labels_to_text(y[i].tolist())
                    total_wer += word_error_rate(gt_text, pred_text)
                    count += 1
        
        return total_wer / count
    


