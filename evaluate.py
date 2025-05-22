import os
import torch
from PIL import Image
from torchvision import transforms
from diffusers import DDPMPipeline, DDPMScheduler
import lpips
import pandas as pd

# ------------------------------
# CONFIG
# ------------------------------
INPUT_DIR = "celebahq_256"
NUM_IMAGES = 2000
NOISE_STEP = 300
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

image_count = 0
for subdir in sorted(os.listdir(INPUT_DIR)):
    subdir_path = os.path.join(INPUT_DIR, subdir)
    if not os.path.isdir(subdir_path):
        continue
    for fname in sorted(os.listdir(subdir_path)):
        if not fname.endswith(".jpg"):
            continue
        img_path = os.path.join(subdir_path, fname)
        try:
            x_0 = preprocess_image(img_path)
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
            continue

        noise = torch.randn_like(x_0)
        x_t = pipe.scheduler.add_noise(x_0, noise, t)

        x_curr = x_t.clone()
        for t_denoise in timesteps[step_index:]:
            with torch.no_grad():
                noise_pred = pipe.unet(x_curr, t_denoise.unsqueeze(0)).sample
                x_curr = pipe.scheduler.step(noise_pred, t_denoise, x_curr).prev_sample

        metrics = calculate_metrics(x_0, x_curr)
        metrics["image_name"] = fname
        results.append(metrics)

        print(f"Finished image: {fname}")  

        image_count += 1
        if image_count >= NUM_IMAGES:
            break
    if image_count >= NUM_IMAGES:
        break

# ------------------------------
# SAVE RESULTS
# ------------------------------
results_df = pd.DataFrame(results)
results_df = results_df[["image_name", "lpips", "psnr", "cos_sim"]]
results_df.to_csv("celebahq_dif_metrics.csv", index=False)
