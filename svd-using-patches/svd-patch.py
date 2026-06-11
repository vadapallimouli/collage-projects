# Google Colab
# Patch-based SVD denoising
# Upload ONE CLEAN image
# Creates synthetic noisy image, denoises it, and gives a clearer final result

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from google.colab import files

uploaded = files.upload()
filename = next(iter(uploaded.keys()))

RESIZE_TO = (256, 256)
clean_img = Image.open(filename).convert("RGB").resize(RESIZE_TO, Image.Resampling.LANCZOS)
clean = np.asarray(clean_img, dtype=np.float32) / 255.0

# Less aggressive synthetic noise so final output stays clearer
NOISE_STD = 0.16
RANDOM_SEED = 7

rng = np.random.default_rng(RANDOM_SEED)
noise = rng.normal(0.0, NOISE_STD, clean.shape).astype(np.float32)
noisy = np.clip(clean + noise, 0.0, 1.0)

def mse(a, b):
    return float(np.mean((a - b) ** 2))

def psnr(gt, img):
    m = mse(gt, img)
    if m < 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / m))

def noise_reduction_percent(clean_img, noisy_img, denoised_img):
    noisy_err = mse(clean_img, noisy_img)
    denoised_err = mse(clean_img, denoised_img)
    return float(((noisy_err - denoised_err) / noisy_err) * 100.0)

def get_positions(length, patch_size, stride):
    if length <= patch_size:
        return [0]
    pos = list(range(0, length - patch_size + 1, stride))
    if pos[-1] != length - patch_size:
        pos.append(length - patch_size)
    return pos

def extract_patches(channel, patch_size, stride):
    h, w = channel.shape
    ys = get_positions(h, patch_size, stride)
    xs = get_positions(w, patch_size, stride)

    patches = []
    positions = []
    for y in ys:
        for x in xs:
            patches.append(channel[y:y+patch_size, x:x+patch_size].reshape(-1))
            positions.append((y, x))
    return np.array(patches, dtype=np.float32), np.array(positions, dtype=np.int32)

def normalize_patch(p):
    mn, mx = p.min(), p.max()
    if mx - mn < 1e-8:
        return np.zeros_like(p)
    return (p - mn) / (mx - mn)

def make_patch_montage(patches, patch_size=8, max_show=100):
    patches = patches[:max_show]
    n = len(patches)
    grid = int(np.ceil(np.sqrt(n)))
    canvas = np.zeros((grid * patch_size, grid * patch_size), dtype=np.float32)

    for i in range(n):
        r = i // grid
        c = i % grid
        patch = normalize_patch(patches[i].reshape(patch_size, patch_size))
        y0 = r * patch_size
        x0 = c * patch_size
        canvas[y0:y0+patch_size, x0:x0+patch_size] = patch

    return canvas

def colorize_channel(channel, color):
    out = np.zeros((channel.shape[0], channel.shape[1], 3), dtype=np.float32)
    if color == "red":
        out[:, :, 0] = channel
    elif color == "green":
        out[:, :, 1] = channel
    elif color == "blue":
        out[:, :, 2] = channel
    return np.clip(out, 0.0, 1.0)

def find_similar_patch_indices(ref_idx, patches, positions, search_radius, max_similar):
    ref_pos = positions[ref_idx]
    ref_patch = patches[ref_idx]

    dy = np.abs(positions[:, 0] - ref_pos[0])
    dx = np.abs(positions[:, 1] - ref_pos[1])
    mask = (dy <= search_radius) & (dx <= search_radius)

    candidate_indices = np.where(mask)[0]
    candidate_patches = patches[candidate_indices]

    dists = np.mean((candidate_patches - ref_patch) ** 2, axis=1)
    order = np.argsort(dists)
    return candidate_indices[order[:max_similar]]

def svd_denoise_group(group, threshold_factor):
    mean_vec = group.mean(axis=0, keepdims=True)
    centered = group - mean_vec

    U, S, VT = np.linalg.svd(centered, full_matrices=False)

    if len(S) == 0:
        return group.copy()

    noise_level = np.median(S) if len(S) > 1 else S[0]
    threshold = threshold_factor * noise_level
    S_new = np.maximum(S - threshold, 0.0)

    # Keep more components to preserve clarity
    if np.count_nonzero(S_new) == 0:
        keep = min(6, len(S))
        S_new[:keep] = 0.85 * S[:keep]

    recon = (U * S_new) @ VT + mean_vec
    return recon.astype(np.float32)

def reconstruct_channel(channel, patch_size, stride, search_radius, max_similar, threshold_factor, collect=False):
    patches, positions = extract_patches(channel, patch_size, stride)

    h, w = channel.shape
    out = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    shown_before = []
    shown_after = []

    for ref_idx in range(len(patches)):
        idxs = find_similar_patch_indices(ref_idx, patches, positions, search_radius, max_similar)
        group = patches[idxs]
        denoised_group = svd_denoise_group(group, threshold_factor)

        ref_in_group = np.where(idxs == ref_idx)[0][0]
        denoised_patch = denoised_group[ref_in_group].reshape(patch_size, patch_size)

        y, x = positions[ref_idx]
        out[y:y+patch_size, x:x+patch_size] += denoised_patch
        weight[y:y+patch_size, x:x+patch_size] += 1.0

        if collect and len(shown_before) < 100:
            shown_before.append(patches[ref_idx])
            shown_after.append(denoised_group[ref_in_group])

    recon = out / np.maximum(weight, 1e-8)
    recon = np.clip(recon, 0.0, 1.0)

    result = {"reconstructed": recon}

    if collect:
        center_idx = len(patches) // 2
        center_ids = find_similar_patch_indices(center_idx, patches, positions, search_radius, max_similar)
        center_group = patches[center_ids]
        center_denoised_group = svd_denoise_group(center_group, threshold_factor)
        center_pos = np.where(center_ids == center_idx)[0][0]

        result.update({
            "channel_gray": channel,
            "patches_before": make_patch_montage(np.array(shown_before), patch_size=patch_size),
            "patches_after": make_patch_montage(np.array(shown_after), patch_size=patch_size),
            "example_before": normalize_patch(patches[center_idx].reshape(patch_size, patch_size)),
            "example_after": normalize_patch(center_denoised_group[center_pos].reshape(patch_size, patch_size)),
        })

    return result

def run_patch_svd(image, patch_size, stride, search_radius, max_similar, threshold_factor, collect=False):
    r = reconstruct_channel(image[:, :, 0], patch_size, stride, search_radius, max_similar, threshold_factor, collect=collect)
    g = reconstruct_channel(image[:, :, 1], patch_size, stride, search_radius, max_similar, threshold_factor, collect=collect)
    b = reconstruct_channel(image[:, :, 2], patch_size, stride, search_radius, max_similar, threshold_factor, collect=collect)

    final_img = np.stack(
        [r["reconstructed"], g["reconstructed"], b["reconstructed"]],
        axis=2
    )
    final_img = np.clip(final_img, 0.0, 1.0)
    return final_img, r, g, b

# Tuned for clarity
best_cfg = {
    "patch_size": 8,
    "stride": 4,
    "search_radius": 12,
    "max_similar": 16,
    "threshold_factor": 0.72
}

best_img, r_res, g_res, b_res = run_patch_svd(noisy, **best_cfg, collect=True)

noisy_psnr = psnr(clean, noisy)
final_psnr = psnr(clean, best_img)
final_reduction = noise_reduction_percent(clean, noisy, best_img)

print("========== FINAL TRUE METRICS ==========")
print(f"Best Parameters            : {best_cfg}")
print(f"Noise Reduced              : {final_reduction:.2f}%")
print(f"PSNR of Noisy Image        : {noisy_psnr:.2f} dB")
print(f"PSNR of Denoised Image     : {final_psnr:.2f} dB")
print(f"PSNR Improvement           : {final_psnr - noisy_psnr:.2f} dB")

R_col = colorize_channel(r_res["channel_gray"], "red")
G_col = colorize_channel(g_res["channel_gray"], "green")
B_col = colorize_channel(b_res["channel_gray"], "blue")

R_rec = colorize_channel(r_res["reconstructed"], "red")
G_rec = colorize_channel(g_res["reconstructed"], "green")
B_rec = colorize_channel(b_res["reconstructed"], "blue")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(clean)
axes[0].set_title("Clean Image", fontsize=15, fontweight="bold")
axes[0].axis("off")

axes[1].imshow(noisy)
axes[1].set_title("Synthetic Noisy Image", fontsize=15, fontweight="bold")
axes[1].axis("off")

axes[2].imshow(best_img)
axes[2].set_title(f"Denoised Image\n({final_reduction:.2f}% Noise Reduction)", fontsize=15, fontweight="bold")
axes[2].axis("off")

plt.tight_layout()
plt.show()

panels = [
    ("Noisy Image", noisy),
    ("R Channel Image", R_col),
    ("G Channel Image", G_col),
    ("B Channel Image", B_col),

    ("R Channel Patches", r_res["patches_before"]),
    ("G Channel Patches", g_res["patches_before"]),
    ("B Channel Patches", b_res["patches_before"]),
    ("Example Patch Before Noise Removal", r_res["example_before"]),

    ("Example Patch After SVD", r_res["example_after"]),
    ("R Denoised Patches After SVD", r_res["patches_after"]),
    ("G Denoised Patches After SVD", g_res["patches_after"]),
    ("B Denoised Patches After SVD", b_res["patches_after"]),

    ("Reconstructed R Channel", R_rec),
    ("Reconstructed G Channel", G_rec),
    ("Reconstructed B Channel", B_rec),
    (f"Final Reconstructed Image\nNoise Reduced: {final_reduction:.2f}%\nPSNR: {final_psnr:.2f} dB", best_img),
]

fig, axes = plt.subplots(4, 4, figsize=(18, 18))
axes = axes.ravel()

for ax, (title, data) in zip(axes, panels):
    if data.ndim == 3:
        ax.imshow(data)
    else:
        ax.imshow(data, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.show()
