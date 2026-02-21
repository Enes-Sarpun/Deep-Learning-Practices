"""
Still_Transfer module.
"""
# Import libraries;
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Note;
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Image;
def Load_Image(image_path, max_size=400, shape=None):
    """"
    Image Path mean is a string path to the image file.
    Max_size is the maximum size of the image.
    Shape is the shape to resize the image to.
    """
    image = Image.open(image_path).convert('RGB')
    
    # Sizing;
    if shape is not None:
        size = shape # Still and Content images should be the same size for style transfer.
    else:
        size = max(image.size) # Take the long side.
        if size > max_size: # If it's too big, make it smaller.
            size = max_size
    
    # Transformations;
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image.to(device)

    # Adjust the image on the screen;
def im_convert(tensor):
    """
    Matplotlib ile görselleştirme yapabilmek için görseli 0-1 aralığına ve (H,W,3) formatına dönüştürmeliyiz.
    """
    image = tensor.clone().detach().cpu().squeeze(0)
    # Multiply by the std and add the mean, in other words, reverse the normalization process.
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    image = image.clamp(0,1) #Clamp to the range 0-1.
    return image.permute(1,2,0).numpy() # (3,H,W) -> (H,W,3)

# The Gram matrix, that is, the style similarity measure;
def Gram_Matrix(tensor):
    """
    (C, H, W) -> (C,H*W) = A,AxA.T formatındaki tensorün Gram matrisini hesaplar.
    """
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)  # (C, H*W)
    gram = torch.mm(tensor, tensor.t())  # (C, C)
    return gram

# Feature extraction model;
class VGG_Feature(nn.Module):
    """
    Feature extraction using the VGG19 model pretrained on ImageNet. (For both content and style)
    """
    def __init__(self):
        super(VGG_Feature, self).__init__()

        # Only 29 Layers;
        self.vgg = models.vgg19(pretrained=True).features[:29].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False # Keep the weights constant.
        # Layer naming;
        self.layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",  # Content feature layer.
            "28": "conv5_1"
        }

    def forward(self, x):
        feature = {}
        for name, layer in self.vgg.named_children():
            x = layer(x)
            if name in self.layers:
                feature[self.layers[name]] = x
        return feature
    
#Complete the Still Transfer Cycle;
def run_style_transfer(content_image,
                       style_image,# Still image.
                       step = 2000,# Iteration number.
                       style_weight = 1e6,# Still loss coefficient.
                       content_weight = 1):# Content loss coefficient.

    # Copy the target image and make it ready for optimization.
        target = content_image.clone().requires_grad_(True).to(device)
        optimizer = optim.Adam([target], lr=0.003)
        model = VGG_Feature()

        for step in tqdm(range(step)):
            target_feature = model(target)
            content_feature = model(content_image)
            style_feature = model(style_image)

        # Content Loss;
        content_loss = torch.mean((target_feature['conv4_2'] - content_feature['conv4_2'])**2)

        # Still loss: Calculate the gram matrix for each selected layer and sum the loss;
        style_loss = 0
        for layer in ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']:
            target_feature_map = target_feature[layer]
            style_feature_map = style_feature[layer]
            target_gram = Gram_Matrix(target_feature_map)
            style_gram = Gram_Matrix(style_feature_map)
            layer_style_loss = torch.mean((target_gram - style_gram)**2)
            style_loss += layer_style_loss

        # Total Loss;
        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print('Step [{}/{}], Total Loss: {:.4f}'.format(step, step, total_loss.item()))
        return target

# Let's apply it;
content = Load_Image("cat.jpg")
style = Load_Image("style.jpg", shape=tuple(content.shape[-2:]))
output = run_style_transfer(content,style)

# Visualize the result;
plt.figure(figsize=(10,5))
plt.imshow(im_convert(output))
plt.title("Still Transfer Sonucu", fontsize=20)
plt.axis('off')
plt.show()



# Finished.