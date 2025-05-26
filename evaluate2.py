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
df = pd.read_csv("top_2_percent_similarity_analysis.csv")
images = df.iloc[:, 0].tolist()

INPUT_DIR = "celebahq_256/celeba_hq_256/"
STEPS = 1000
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESIZE_SIZE = 256
NOISE_STEPS = [330, 380, 420]

# ------------------------------
# SETUP
# ------------------------------
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
            original.flatten(), 
            reconstructed.flatten(), 
            dim=0
        ).item()
    }

def save_tensor_image(tensor, path):
    img = tensor.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0) * 0.5 + 0.5 
    img = img.clamp(0, 1).numpy()
    Image.fromarray((img * 255).astype("uint8")).save(path)

# ------------------------------
# MAIN LOOP: Over Noise Levels
# ------------------------------
for NOISE_STEP in NOISE_STEPS:
    print(f"\n=== Running for NOISE_STEP = {NOISE_STEP} ===")
    pipe.scheduler.set_timesteps(STEPS, device=DEVICE)
    timesteps = pipe.scheduler.timesteps
    step_index = len(timesteps) - 1 - NOISE_STEP
    t = timesteps[step_index].unsqueeze(0)

    output_dir = f"recon_vis/noise_{NOISE_STEP}"
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for img_name in images:
        img_path = os.path.join(INPUT_DIR, img_name)
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

        base_name = os.path.splitext(os.path.basename(img_name))[0]
        save_tensor_image(x_0, f"{output_dir}/{base_name}_original.png")
        save_tensor_image(x_t, f"{output_dir}/{base_name}_noised.png")
        save_tensor_image(x_curr, f"{output_dir}/{base_name}_reconstructed.png")

        metrics = calculate_metrics(x_0, x_curr)
        metrics["image_name"] = img_name
        metrics["noise_step"] = NOISE_STEP
        results.append(metrics)

        print(f"✓ {img_name} (noise={NOISE_STEP})")

    # ------------------------------
    # SAVE METRICS
    # ------------------------------
    df_out = pd.DataFrame(results)
    df_out = df_out[["image_name", "noise_step", "lpips", "psnr", "cos_sim"]]
    df_out.to_csv(f"celebahq_dif_metrics_noise_{NOISE_STEP}.csv", index=False)
    print(f"📁 Saved CSV for noise {NOISE_STEP}")
