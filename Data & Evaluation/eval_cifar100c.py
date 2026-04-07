import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(PROJECT_ROOT)

from models.composite_vit import get_composite_vit
from models.vpt import get_vpt_vit

HIGH_FREQ_CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise', 'spatter']
LOW_FREQ_CORRUPTIONS = ['snow', 'frost', 'fog', 'brightness', 'contrast']
SPATIAL_CORRUPTIONS = ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'elastic_transform']
CORRUPTIONS = HIGH_FREQ_CORRUPTIONS + LOW_FREQ_CORRUPTIONS + SPATIAL_CORRUPTIONS


def evaluate_cifar100c(model, device, data_dir, batch_size=64):
    model.eval()
    results = {}

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    labels_path = os.path.join(data_dir, 'labels.npy')
    labels = np.load(labels_path)

    for corruption in CORRUPTIONS:
        data_path = os.path.join(data_dir, f'{corruption}.npy')
        if not os.path.exists(data_path):
            print(f"Warning: Data for {corruption} not found at {data_path}.")
            continue

        data = np.load(data_path)
        correct = 0
        total = 0

        class CifarC_Dataset(torch.utils.data.Dataset):
            def __init__(self, data_array, label_array, transform):
                self.data = data_array
                self.labels = label_array
                self.transform = transform

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                img = self.data[idx]
                label = self.labels[idx]
                return self.transform(img), torch.tensor(label, dtype=torch.long)

        dataset = CifarC_Dataset(data, labels, transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        progress = tqdm(loader, desc=f"Evaluating {corruption}")
        with torch.no_grad():
            for inputs, targets in progress:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        acc = 100.0 * correct / total
        results[corruption] = acc
        print(f"[{corruption}] Accuracy: {acc:.2f}%")

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "./data/CIFAR-100-C"
    checkpoint_dir = "./outputs"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    variants_to_test = {
        "VPT_Deep": ("vpt", 16, 0.0, False),
        "LoRA_Only": ("lora_only", 16, 0.0, False),
        "DoRA_Only": ("dora_only", 16, 0.0, True),
        "SS_Adapter_LoRA": ("lora_composite", 16, 1e-4, False),
        "SS_Adapter_DoRA": ("dora_composite", 16, 1e-4, True),
    }

    final_report = []

    for name, (variant, r, init_scale, use_dora) in variants_to_test.items():
        print(f"\n{'=' * 50}\nTesting: {name}\n{'=' * 50}")

        if variant == "vpt":
            model = get_vpt_vit(num_prompts=r).to(device)
        else:
            model = get_composite_vit(lora_r=r, use_dora=use_dora, init_scale=init_scale).to(device)

        import glob
        pattern = os.path.join(checkpoint_dir, f"*{variant}*", "best.pt")
        ckpt_paths = glob.glob(pattern)

        if not ckpt_paths:
            print(f"Warning: Checkpoint not found for {name}, skipping...")
            continue

        checkpoint = torch.load(ckpt_paths[0], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"Loaded weights from: {ckpt_paths[0]}")

        res = evaluate_cifar100c(model, device, data_dir)

        high_freq_acc = np.mean([res[c] for c in HIGH_FREQ_CORRUPTIONS])
        low_freq_acc = np.mean([res[c] for c in LOW_FREQ_CORRUPTIONS])
        spatial_acc = np.mean([res[c] for c in SPATIAL_CORRUPTIONS])
        mean_acc = np.mean(list(res.values()))

        res.update({
            "Method": name,
            "Mean_Acc": mean_acc,
            "High_Freq_Acc": high_freq_acc,
            "Low_Freq_Acc": low_freq_acc,
            "Spatial_Acc": spatial_acc
        })
        final_report.append(res)

        single_df = pd.DataFrame([res])
        single_csv_path = os.path.join(output_dir, f"{name}_details.csv")
        single_df.to_csv(single_csv_path, index=False)

    df = pd.DataFrame(final_report)
    cols = ['Method', 'Mean_Acc', 'High_Freq_Acc', 'Low_Freq_Acc', 'Spatial_Acc'] + CORRUPTIONS
    df = df[cols]

    final_csv_path = os.path.join(output_dir, "Table3_CIFAR100C_Robustness.csv")
    df.to_csv(final_csv_path, index=False)

    print(f"\nEvaluation completed. Results saved to: {output_dir}")
    print(f"Summary table: {final_csv_path}")


if __name__ == "__main__":
    main()