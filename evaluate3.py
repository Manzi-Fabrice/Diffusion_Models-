
import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from diffusers import DDPMPipeline, DDPMScheduler
import lpips
import pandas as pd

# ------------------------------ CONFIG ------------------------------
IMAGE_DIR = "celebahq_256/celeba_hq_256"
IMAGE_LIST_FILE = "combined_image_list.txt"
NOISE_STEPS = [300, 350, 400, 450, 500]
STEPS = 1000
RESIZE_SIZE = 256
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------ SETUP ------------------------------
torch.manual_seed(SEED)
pipe = DDPMPipeline.from_pretrained("google/ddpm-ema-celebahq-256")
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(DEVICE)
pipe.unet.eval()
lpips_model = lpips.LPIPS(net='alex').to(DEVICE)

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
            original.flatten(), reconstructed.flatten(), dim=0
        ).item()
    }

# ------------------------------ LOAD IMAGE LIST ------------------------------
with open(IMAGE_LIST_FILE, "r") as f:
    image_list = [line.strip() for line in f if line.strip()]

# ------------------------------ RUN LOOP ------------------------------
original_out = "originals"
os.makedirs(original_out, exist_ok=True)

for NOISE_STEP in NOISE_STEPS:
    print(f"\n🚀 Running inversion at NOISE_STEP={NOISE_STEP}...")
    pipe.scheduler.set_timesteps(STEPS, device=DEVICE)
    timesteps = pipe.scheduler.timesteps
    step_index = len(timesteps) - 1 - NOISE_STEP
    t = timesteps[step_index].unsqueeze(0)

    results = []
    recon_dir = f"recons/{NOISE_STEP}"
    os.makedirs(recon_dir, exist_ok=True)

    for fname in image_list:
        img_path = os.path.join(IMAGE_DIR, fname)
        try:
            x_0 = preprocess_image(img_path)
        except Exception as e:
            print(f"⚠️ Failed to load {img_path}: {e}")
            continue

        if NOISE_STEP == NOISE_STEPS[0]:
            save_image(x_0, os.path.join(original_out, fname))

        noise = torch.randn_like(x_0)
        x_t = pipe.scheduler.add_noise(x_0, noise, t)
        x_curr = x_t.clone()

        for t_denoise in timesteps[step_index:]:
            with torch.no_grad():
                noise_pred = pipe.unet(x_curr, t_denoise.unsqueeze(0)).sample
                x_curr = pipe.scheduler.step(noise_pred, t_denoise, x_curr).prev_sample

        save_image(x_curr, os.path.join(recon_dir, fname))
        metrics = calculate_metrics(x_0, x_curr)
        metrics["image_name"] = fname
        metrics["noise_step"] = NOISE_STEP
        results.append(metrics)

        print(f"✅ Done: {fname} @ noise {NOISE_STEP}")

    df = pd.DataFrame(results)
    df = df[["image_name", "noise_step", "lpips", "psnr", "cos_sim"]]
    df.to_csv(f"celebahq_metrics_{NOISE_STEP}.csv", index=False)
    print(f"📁 Saved metrics to celebahq_metrics_{NOISE_STEP}.csv")
