import torch
from PIL import Image
from torchvision import transforms
from diffusers import DDPMPipeline, DDPMScheduler
import os
import numpy as np
import lpips

# ------------------------------
# CONFIG
# ------------------------------
SAVE_DIR = "ddpm_inversion_results"
NOISE_STEP = 300  # Recommended starting point for CelebA-HQ
STEPS = 1000
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATH = "celebahq_256/celeba_hq_256/00000.jpg" 
RESIZE_SIZE = 256

# ------------------------------
# SETUP
# ------------------------------
os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(SEED)

# Load model with correct DDPM scheduler
pipe = DDPMPipeline.from_pretrained("google/ddpm-ema-celebahq-256")
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(DEVICE)
pipe.unet.eval()

# LPIPS model
lpips_model = lpips.LPIPS(net='alex').to(DEVICE)

# ------------------------------
# IMAGE PROCESSING FUNCTIONS
# ------------------------------
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(RESIZE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Scales to [-1, 1]
    ])
    return transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)

def tensor_to_image(tensor):
    """Convert model output tensor to valid PIL Image"""
    img = tensor.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0) * 0.5 + 0.5  # Scale to [0, 1]
    img = img.clamp(0, 1).numpy()  # Clamp to valid range
    return Image.fromarray((img * 255).astype(np.uint8))

# ------------------------------
# IMAGE LOADING
# ------------------------------
try:
    x_0 = preprocess_image(IMAGE_PATH)
    print(f" Loaded image: {IMAGE_PATH}")
except Exception as e:
    print(f" Error loading image: {e}, using random tensor")
    x_0 = torch.randn(1, 3, RESIZE_SIZE, RESIZE_SIZE, device=DEVICE)

# Save original image
tensor_to_image(x_0).save(f"{SAVE_DIR}/original.png")

# ------------------------------
# NOISE INJECTION
# ------------------------------
pipe.scheduler.set_timesteps(STEPS, device=DEVICE)
timesteps = pipe.scheduler.timesteps

# Calculate step index (reverse order: timesteps[0] = 999)
step_index = len(timesteps) - 1 - NOISE_STEP
t = timesteps[step_index].unsqueeze(0)

# Add noise using DDPM's forward process
noise = torch.randn_like(x_0)
x_t = pipe.scheduler.add_noise(x_0, noise, t)

# Save noisy image
tensor_to_image(x_t).save(f"{SAVE_DIR}/noisy.png")

# ------------------------------
# DENOISING PROCESS
# ------------------------------
x_curr = x_t.clone()
denoise_timesteps = timesteps[step_index:]

for t in denoise_timesteps:
    with torch.no_grad():
        noise_pred = pipe.unet(x_curr, t.unsqueeze(0)).sample
        x_curr = pipe.scheduler.step(noise_pred, t, x_curr).prev_sample

# ------------------------------
# METRICS & SAVING
# ------------------------------
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

# Save final reconstruction
tensor_to_image(x_curr).save(f"{SAVE_DIR}/reconstructed.png")

# Calculate and print metrics
metrics = calculate_metrics(x_0, x_curr)
print("\n📊 Evaluation Metrics:")
print(f"LPIPS: {metrics['lpips']:.4f} (Lower is better)")
print(f"PSNR: {metrics['psnr']:.2f} dB (Higher is better)")
print(f"Cosine Similarity: {metrics['cos_sim']:.4f}")

print(f"\n✅ Results saved to {SAVE_DIR}/")