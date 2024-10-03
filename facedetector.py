import cv2
import numpy as np
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
import onnxruntime
from scipy.spatial.distance import cosine
import torch
from torchvision.transforms import ToPILImage

class FaceSwap:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.face_analyser = self.get_face_analyser()
        self.face_swapper = self.get_face_swap_model()

    def get_face_analyser(self, providers=None, det_size=(320, 320)):
        face_analyser = FaceAnalysis(providers=providers)
        face_analyser.prepare(ctx_id=0, det_size=det_size)
        return face_analyser

    def get_face_swap_model(self):
        model = insightface.model_zoo.get_model(self.model_path)
        return model

    def get_one_face(self, frame: np.ndarray):
        faces = self.face_analyser.get(frame)
        try:
            return min(faces, key=lambda x: x.bbox[0])
        except ValueError:
            return None

   def extract_embedding(self, tensor: torch.Tensor):
       to_pil = ToPILImage()
       if tensor.dim() == 4:
           tensor = tensor.squeeze(0)
       image = to_pil(tensor.cpu())
       face = self.get_one_face(np.array(image))
       if face is not None:
           embedding = face.embedding
           if isinstance(embedding, np.ndarray):
               embedding = torch.tensor(embedding)
           return face.embedding
       return None
    # def extract_embedding(self, tensor: torch.Tensor):
    #     if tensor.dim() == 4:
    #         tensor = tensor.squeeze(0)
    
    #     # detach()로 기울기 추적을 중단
    #     face = self.get_one_face(tensor.permute(1, 2, 0).cpu().detach().numpy())  

    #     if face is not None:
    #         embedding = face.embedding

    #         if isinstance(embedding, np.ndarray):
    #             embedding = torch.tensor(embedding, device=tensor.device)

    #         return embedding
    #     return None



    def swap_face(self, target_face, source_face_embedding, temp_frame):
        class SourceFace:
            def __init__(self, embedding):
                self.normed_embedding = embedding.astype(np.float32)

        source_face = SourceFace(source_face_embedding)
        result = self.face_swapper.get(temp_frame, target_face, source_face, paste_back=True)
        return result

    def swap_faces(self, source_img_path: str, target_img_path: str):
        source_image = Image.open(source_img_path)
        target_image = Image.open(target_img_path)

        target_img = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)
        target_face = self.get_one_face(target_img)

        if target_face is not None:
            source_img = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)
            source_embedding = self.extract_embedding(source_img)

            if source_embedding is not None:
                temp_frame = target_img.copy()
                temp_frame = self.swap_face(target_face, source_embedding, temp_frame)

                similarity = 1 - cosine(source_embedding, target_face.embedding)
                print(f"Cosine similarity between source and target embeddings: {similarity}")

                result_image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
                result_image.save("result.png")
                print("Result saved as result.png")
            else:
                print("No faces found in source image!")
        else:
            print("No target faces found!")

