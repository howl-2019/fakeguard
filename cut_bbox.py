import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceAligner:
    def __init__(self, det_size=(320, 320)):
        self.face_analyser = self.get_face_analyser(det_size)

    def get_face_analyser(self, det_size):
        face_analyser = FaceAnalysis()
        face_analyser.prepare(ctx_id=0, det_size=det_size)
        return face_analyser

    def get_face_bbox(self, image):
        # 얼굴 감지 및 바운딩 박스 가져오기
        faces = self.face_analyser.get(image)
        if len(faces) == 0:
            return None
        # 가장 큰 얼굴을 기준으로 함
        return max(faces, key=lambda x: x.bbox[2] * x.bbox[3])

    def crop_face_bbox(self, image, face):
        # 얼굴 바운딩 박스 부분만 자르기
        x1, y1, x2, y2 = face.bbox.astype(int)
        cropped_face = image[y1:y2, x1:x2]
        return cropped_face

def process_images(source_img_path, target_img_path):
    # 이미지 불러오기
    source_image = cv2.imread(source_img_path)
    target_image = cv2.imread(target_img_path)

    # 얼굴 정렬 클래스 생성
    aligner = FaceAligner()

    # 소스 및 타겟 이미지에서 얼굴 감지
    source_face = aligner.get_face_bbox(source_image)
    target_face = aligner.get_face_bbox(target_image)

    if source_face is None or target_face is None:
        print("One of the images does not contain a detectable face.")
        return None

    # 얼굴 바운딩 박스 부분만 자르기
    cropped_source_face = aligner.crop_face_bbox(source_image, source_face)
    cropped_target_face = aligner.crop_face_bbox(target_image, target_face)

    # 이미지 저장
    cv2.imwrite("analysis_result/face_eps_0.08.png", cropped_source_face)
    cv2.imwrite("analysis_result/face_eps_0.1.png", cropped_target_face)
    print("Cropped faces saved.")

if __name__ == "__main__":
    # 소스와 타겟 이미지 경로를 입력
    process_images("analysis_result/eps_0.08.png", "analysis_result/eps_0.1.png")