import csv
import zipfile
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.colors import ListedColormap

gradient_cmaps = ['viridis', 'plasma', 'Greens', 'YlGnBu', 'PiYG']
categorical_cmaps = ['tab10', 'Set3', 'tab20', 'jet', 'Paired']

os.makedirs("color_palettes", exist_ok=True)

def save_cmap_as_csv(name, cmap, num_colors=256):
    with open(f"color_palettes/{name}.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["r", "g", "b"])
        for i in range(num_colors):
            r, g, b, _ = cmap(i / (num_colors - 1))
            writer.writerow([int(r * 255), int(g * 255), int(b * 255)])

def display_cmap(cmap, name):
    gradient = np.linspace(0, 1, 256).reshape(1, 256)
    gradient = np.vstack((gradient, gradient))

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.set_axis_off()
    ax.set_title(name)
    plt.show()

# Save gradient palettes
for cmap_name in gradient_cmaps:
    cmap = plt.get_cmap(cmap_name)
    save_cmap_as_csv(cmap_name, cmap)
    display_cmap(cmap, cmap_name)

# Save categorical palettes (with fewer colors)
for cmap_name in categorical_cmaps:
    cmap = plt.get_cmap(cmap_name)
    save_cmap_as_csv(cmap_name, cmap, num_colors=500)
    display_cmap(cmap, cmap_name)

# Create a ZIP file
zip_filename = "color_palettes_csv.zip"
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for filename in os.listdir("color_palettes"):
        zipf.write(os.path.join("color_palettes", filename), arcname=filename)

# Download
from google.colab import files
files.download(zip_filename)
