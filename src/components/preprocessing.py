import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch 

class Tokenizer():
    def __init__(self):
        self.vocab = list("abcdefghijklmnopqrstuvwxyz1234567890!ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
        self.char_to_idx = {c: i+1 for i, c in enumerate(self.vocab)} 
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
    def text_to_labels(self,text):
        output=[]
        for c in text:
            output.append(self.char_to_idx[c])
        return output



def decode_align(path):
    words=[]
    with open(path,'r') as fobj:
        for line in fobj:
            _,_,word=line.strip().split()
            if word!= "sil":
                words.append(word)
    return " ".join(words)


class Frames():
    def __init__(self,model_path):
        base_options = python.BaseOptions(
            model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)


        # MediaPipe lip landmark indices
        self.LIPS = [
            61, 146, 91, 181, 84, 17, 314, 405,
            321, 375, 291, 308, 324, 318, 402,
            317, 14, 87, 178, 88, 95, 185,
            40, 39, 37, 0, 267, 269, 270,
            409, 415, 310, 311, 312, 13,
            82, 81, 42, 183, 78
        ]


    def extract_mouth_frames(self,video_path):
        cap = cv2.VideoCapture(video_path)

        mouth_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
        
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )
        
            result = self.detector.detect(mp_image)
        
            if not result.face_landmarks:
                continue
        
            landmarks = result.face_landmarks[0]
        
            xs = [int(landmarks[i].x * w) for i in self.LIPS]
            ys = [int(landmarks[i].y * h) for i in self.LIPS]
        
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
        
            mouth = frame[y_min:y_max, x_min:x_max]
        
            
            mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
            mouth = cv2.resize(mouth, (100, 50))
        
            mouth_frames.append(mouth)
        
        cap.release()
        

        return mouth_frames



class Loader():
    def __init__(self):
        pass
    def load_data(self,X_path,y_path, alignment_data_path, speaker_data_path,landmark_path):

        X_all = []
        y_all = []

        for align_file, video_file in zip(X_path, y_path):
            alignment_path=os.path.join(alignment_data_path,align_file)
            speaker_path=os.path.join(speaker_data_path,video_file)
            
            frames=Frames(landmark_path)
            tokeniser=Tokenizer()

            text=decode_align(alignment_path)
            decode_text=tokeniser.text_to_labels(text)

                    
            frames=frames.extract_mouth_frames(speaker_path)
            frames = torch.tensor(frames, dtype=torch.float32)
            frames = frames.unsqueeze(0)

            X_all.append(frames)
            y_all.append(torch.tensor(decode_text, dtype=torch.long))   

        video_tensors = torch.nn.utils.rnn.pad_sequence(
            X_all, batch_first=True
        )   
                    
        return video_tensors, y_all
        