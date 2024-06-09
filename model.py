import torch
import torch.nn as nn
from torchvision import models

class MyModel(nn.Module):
    def __init__(self, num_classes=3):
        super(MyModel, self).__init__()
        
        # 사전 학습된 DenseNet121 모델 불러오기
        self.densenet = models.densenet121(pretrained=True)
        
        # 마지막 fully connected layer를 새로운 클래스 수에 맞게 조정
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.densenet(x)
        return x