import os
import torch
from PIL import Image
from torchvision import transforms
from diffusers import DDPMPipeline, DDPMScheduler
import lpips
import pandas as pd

# getting the top 2% images 
df = pd.read_csv("top_2_percent_similarity_analysis.csv")
temp_images = df.iloc[:,0]
images = temp_images.tolist()

# ------------------------------
# CONFIG
# ------------------------------
INPUT_DIR = "celebahq_256/celeba_hq_256/" # the path to the image
NUM_IMAGES = 2000
NOISE_STEP = 400
STEPS = 1000
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESIZE_SIZE = 256

# ------------------------------
# SETUP
# ------------------------------
torch.manual_seed(SEED)

pipe = DDPMPipeline.from_pretrained("google/ddpm-ema-celebahq-256")
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(DEVICE)
pipe.unet.eval()

lpips_model = lpips.LPIPS(net='alex').to(DEVICE)

# ------------------------------
# IMAGE PROCESSING
# ------------------------------
transform = transforms.Compose([
    transforms.Resize(RESIZE_SIZE),
    transforms.CenterCrop(RESIZE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def preprocess_image(image_path):
    return transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)

def calculate_metrics(original, reconstructed):
    original = original.clamp(-1, 1)
    reconstructed = reconstructed.clamp(-1, 1)
    return {
        'lpips': lpips_model(original, reconstructed).item(),
        'psnr': 10 * torch.log10(1 / torch.nn.functional.mse_loss(original, reconstructed)).item(),
        'cos_sim': torch.nn.functional.cosine_similarity(
            original.flatten(), 
            reconstructed.flatten(), 
            dim=0
        ).item()
    }

# ------------------------------
# INVERSION LOOP
# ------------------------------
pipe.scheduler.set_timesteps(STEPS, device=DEVICE)
timesteps = pipe.scheduler.timesteps
step_index = len(timesteps) - 1 - NOISE_STEP
t = timesteps[step_index].unsqueeze(0)

results = []


def add_noise(shape , region):
    mask = torch.zeros(shape, device=DEVICE)
    _, _, h, w = shape
    if region == "left":
        mask[:, :, :, :w//2] = 1 
    elif region == "right":
        mask[:, :, :, w//2:] = 1 
    elif region == "top":
        mask[:, :, :h//2, :] = 1  
    elif region == "bottom":
        mask[:, :, h//2:, :] = 1 
    elif region == "center":
        margin = h // 4
        mask[:, :, margin:-margin, margin:-margin] = 1 
    elif region == "center_clear":
        mask = torch.ones(shape, device=DEVICE) 
        margin = h // 4
        mask[:, :, margin:-margin, margin:-margin] = 0  


    return mask

def save_tensor_image(tensor, path):
    """Convert a tensor to a PIL image and save it."""
    img = tensor.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0) * 0.5 + 0.5 
    img = img.clamp(0, 1).numpy()
    Image.fromarray((img * 255).astype("uint8")).save(path)


directions = ["center_clear"]

for direction in directions:
    results = []
    os.makedirs(f"recon_vis/{direction}", exist_ok=True)
    for img in images:
        img_name = os.path.splitext(os.path.basename(img))[0]
        img_path = os.path.join(INPUT_DIR, img)
        x_0 = preprocess_image(img_path)

        noise = torch.randn_like(x_0)
        mask = add_noise(x_0.shape, direction)
        partial_noise = mask * noise 
        x_t = pipe.scheduler.add_noise(x_0, partial_noise, t)
        x_curr = x_t.clone()
        for t_denoise in timesteps[step_index:]:
            with torch.no_grad():
                noise_pred = pipe.unet(x_curr, t_denoise.unsqueeze(0)).sample
                x_curr = pipe.scheduler.step(noise_pred, t_denoise, x_curr).prev_sample

        save_tensor_image(x_0, f"recon_vis/{direction}/{img_name}_original.png")
        save_tensor_image(x_t, f"recon_vis/{direction}/{img_name}_noised.png")
        save_tensor_image(x_curr, f"recon_vis/{direction}/{img_name}_reconstructed.png")

        metrics = calculate_metrics(x_0, x_curr)
        metrics["region"] = direction
        metrics["image_name"] = img_path
        results.append(metrics)

        print(f"Finished image: {img_path}")  
    # ------------------------------
    # SAVE RESULTS
    # ------------------------------
    results_df = pd.DataFrame(results)
    results_df = results_df[["image_name", "lpips", "psnr", "cos_sim"]]
    results_df.to_csv(f"celebahq_dif_metrics_{direction}.csv", index=False)






       
