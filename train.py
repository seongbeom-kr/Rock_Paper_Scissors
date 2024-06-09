import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
import os
from model import MyModel  # model.py에서 MyModel 클래스를 임포트
from collections import Counter

# Argument parser 설정
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True, help="path to input dataset")
parser.add_argument("-m", "--model", type=str, required=True, help="path to save the model parameters")
args = parser.parse_args()
root_dir = args.dataset
save_path = args.model

# 커스텀 데이터셋 클래스 정의
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.labels = self._get_image_paths_and_labels()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def _get_image_paths_and_labels(self):
        image_paths = []
        labels = []
        class_dirs = sorted(os.listdir(self.root_dir))
        
        # 클래스 이름을 숫자로 변환하는 매핑 사전 생성
        class_dirs = [d for d in class_dirs if os.path.isdir(os.path.join(self.root_dir, d)) and d.isdigit()]
        if not class_dirs:
            raise ValueError(f"No valid class directories found in {self.root_dir}")
        label_map = {int(class_name): idx for idx, class_name in enumerate(class_dirs)}
        print(f"Label mapping: {label_map}")  # 라벨 매핑 정보 출력
        
        for class_name in class_dirs:
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                image_files = sorted(os.listdir(class_path))
                for image_file in image_files:
                    image_path = os.path.join(class_path, image_file)
                    if image_file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                        image_paths.append(image_path)
                        labels.append(label_map[int(class_name)])  # 라벨 변환
        return image_paths, labels

# 데이터 전처리 설정
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 이미지를 256x256 크기로 조정
    transforms.RandomHorizontalFlip(),  # 무작위 수평 뒤집기
    transforms.RandomRotation(30),  # 회전 각도를 30도로 증가
    transforms.RandomResizedCrop(256, scale=(0.7, 1.0)),  # 무작위 크롭 및 리사이즈
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # 색상 변화 강화
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
])

# 데이터셋 및 데이터로더 설정
dataset = ImageDataset(root_dir, transform=transform)
if len(dataset) == 0:
    raise ValueError(f"No valid images found in the dataset at {root_dir}")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 클래스 분포 확인
class_distribution = Counter(dataset.labels)
print("Class distribution:", class_distribution)

# 모델 설정
num_classes = 3  # 가위바위보 3개 클래스
model = MyModel(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 손실 함수 및 옵티마이저 설정
# 클래스 분포를 기반으로 가중치 설정 (여기서는 예시로 균등 가중치를 사용, 필요시 수정)
class_weights = torch.FloatTensor([1.0, 1.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습률 스케줄러 설정
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 학습 루프
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device).long()  # 라벨이 정수형인지 확인

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    # 학습률 감소 적용
    scheduler.step()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_predictions / total_predictions
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy*100:.2f}%")

print("Training finished!")

# 학습된 모델 저장
torch.save(model.state_dict(), save_path)
print("Model parameters saved successfully!")
