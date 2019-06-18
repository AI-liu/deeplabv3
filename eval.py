import torch
model = torch.load('deeplabv3.pth')
model.eval()
print(model)

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
#input_image = Image.open("/home/cicv/labelimage0320/labelimage/image/segmentation/1532589942135011547.png")
input_image = Image.open("aachen_000000_000019_leftImg8bit.png")
input_image = input_image.resize((int(input_image.width/2),int(input_image.height/2)),Image.NEAREST)
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)[0]
output_predictions = output.argmax(0)
print(output_predictions)

# create a color pallette, selecting a color for each class
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)

import matplotlib.pyplot as plt
plt.imshow(r)
plt.show()


