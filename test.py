import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model import MyModel
import argparse
import os

# Argument parser 설정
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True, help="path to test dataset")  # 테스트 데이터셋 경로
parser.add_argument("-m", "--model", type=str, required=True, help="path to model parameter")  # 학습된 모델 경로
parser.add_argument("-o", "--output", type=str, required=True, help="path to output directory")  # 출력 파일 경로
args = parser.parse_args()
test_dir = args.dataset
model_path = args.model
output_dir = args.output

# TestDataset 클래스 정의
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_path

    def _get_image_paths(self):
        image_paths = []
        image_files = sorted(os.listdir(self.root_dir))
        for image_file in image_files:
            image_path = os.path.join(self.root_dir, image_file)
            image_paths.append(image_path)
        return image_paths

# 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 이미지를 256x256 크기로 조정
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
])

# 테스트 데이터셋 및 데이터로더 설정
test_dataset = TestDataset(test_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 설정
num_classes = 3  # 분류할 클래스 수
model = MyModel(num_classes=num_classes)

# 장치 설정 (GPU 사용 가능하면 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 학습된 모델 파라미터 로드 (CPU와 GPU 환경 모두 지원)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 예측 수행 및 결과 저장
predictions = []
image_names = []

# 예측된 클래스에 따른 이기는 레이블 맵핑
winning_label = {0: 2, 1: 0, 2: 1}  # 주먹 -> 보, 가위 -> 주먹, 보 -> 가위

with torch.no_grad():
    for images, image_paths in test_dataloader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        winning_predictions = [winning_label[p] for p in predicted.cpu().numpy().tolist()]  # 이기는 레이블로 변환
        predictions.extend(winning_predictions)
        image_names.extend(image_paths)

# 결과를 output.txt에 저장
output_file = os.path.join(output_dir, 'output.txt')  # output.txt 파일 경로 설정
with open(output_file, 'w') as f:
    for name, prediction in zip(image_names, predictions):
        f_name = os.path.basename(name)  # 파일 이름만 추출
        f.write(f"{f_name}\t{prediction}\n")

print(f"Predictions saved to {output_file}")
