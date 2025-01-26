from torchvision import transforms
from torchvision.transforms import ToTensor


class ResizePadTransform:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, image):
        # Calculate the aspect ratio of the original image
        width, height = image.size
        aspect_ratio = width / height

        # Determine the size after resizing while preserving the aspect ratio
        if width > height:
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        else:
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)

        # Define a torchvision transform to resize the image
        resize_transform = transforms.Resize((new_height, new_width))

        # Resize the image using the defined transform
        resized_image = resize_transform(image)

        # Calculate the padding required to achieve the target size
        pad_width = self.target_size - new_width
        pad_height = self.target_size - new_height

        # If pad is odd, then it will have issues, so you need to fix it
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        # Create a padding transform using torchvision
        padding_transform = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom))

        # Apply the padding transform to the resized image
        padded_resized_image = padding_transform(resized_image)

        return padded_resized_image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

tfms = transforms.Compose([
    ResizePadTransform(224),
    ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])