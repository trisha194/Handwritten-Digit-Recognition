from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# Flask app
app = Flask(__name__)

# Define CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model
model = SimpleCNN()
model.load_state_dict(torch.load("mnist_cnn_model.pth", map_location=torch.device('cpu')))
model.eval()

# Preprocessing for input images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if file is uploaded
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            # Open image
            image = Image.open(file).convert("L")
            # Transform image
            input_image = transform(image).unsqueeze(0)
            # Make prediction
            with torch.no_grad():
                output = model(input_image)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted = torch.argmax(probabilities, 1).item()

            # Render result
            return render_template("index.html", prediction=predicted, probabilities=probabilities[0])

    return render_template("index.html", prediction=None)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use $PORT if available, otherwise default to 5000
    app.run(host="0.0.0.0", port=port)


