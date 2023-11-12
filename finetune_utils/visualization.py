import torch
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw
from pathlib import Path

def overlay_mask_on_image(image: torch.Tensor, mask: torch.Tensor, bbox: torch.Tensor, threshold=0.5, save_dir="./images", info=(0,0)):
    """
    Overlay a mask onto an image.

    Args:
    - image (torch.Tensor): The original image as a tensor.
    - mask (torch.Tensor): The mask to be overlaid as a tensor.
    - bbox (torch.Tensor): The bounding box coordinates as a tensor.
    - threshold (float, optional): Threshold value to apply on mask. Defaults to 0.5.
    - save_dir (str, optional): Directory to save the resulting image. Defaults to "./images".
    - info (tuple, optional): Extra information to append to the saved image filename. Defaults to (0, 0).

    Returns:
    - PIL.Image: The image with the mask overlaid.

    This function overlays a given mask on an image. The mask is first thresholded: 
    values above the threshold are set to 1, others set to 0. A colored mask (red in this case)
    is created and added to the original image. The resulting image is clamped to ensure
    pixel values are within acceptable limits (0-1) and then scaled to 0-255.
    The bounding box is drawn and the image is saved to the specified directory.
    """

    # Ensure save directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Filename for saving image
    image_path = save_dir / f'{info[0]}_{info[1]}.jpg'

    # Threshold the mask
    threshold_mask = (mask > threshold).float()

    # Create a red colored mask
    colored_mask = torch.zeros_like(image)
    colored_mask[0] = threshold_mask  # Red channel

    # Overlay the mask onto the image
    combined_image = (image + colored_mask).clamp(0, 1)

    # Scale to 0-255 and convert to PIL Image
    combined_image = (combined_image * 255).type(torch.uint8)
    pil_image = to_pil_image(combined_image)

    # Draw bounding box
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle(bbox.tolist(), outline='red', width=2)

    # Save and return the image
    pil_image.save(image_path)
    return pil_image

if __name__ == '__main__':
    image = torch.randn((1024, 1024))
    mask = torch.randint(0, 2, (1024, 1024)).float()
    result = overlay_mask_on_image(image, mask)
    result.save("overlayed_image.jpg")