###################################################################################################
# Example
# python generate_mask_from_WSIs.py 'slide input directory' 'prediction output directory' 'model path' 'norm stats'
# python generate_mask_from_WSIs.py /home/svs/ /home/predict/ model.h5 reinhardStats.csv
###################################################################################################

import os
import sys
import math
import openslide
import numpy as np
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from openslide.deepzoom import DeepZoomGenerator
from reinhard import reinhard
from torchvision import transforms
import cv2
from segmentation_models_pytorch import Unet
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def slide_prediction(file_path, m_model, s_mu_lab, s_sigma_lab, out_path, s_name, t_size, b_scale):
    try:
        slide = openslide.OpenSlide(file_path)
    except Exception as e:
        return

    (width, height) = slide.level_dimensions[0]
    #generator to efficiently extract tiles from the slide.
    generator = DeepZoomGenerator(slide, tile_size=t_size, overlap=0, limit_bounds=True)
    highest_zoom_level = generator.level_count - 1

    try:
        mag = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        offset = math.floor((mag / 20) / 2)
        # Sets the zoom level for processing based on magnification.
        level = highest_zoom_level - offset
    except KeyError:
        mag = None
        level = highest_zoom_level

    cols, rows = generator.level_tiles[level]
    validate = True
    if mag == 40:
        height_adj = round(height * b_scale)
        width_adj = round(width * b_scale)
        tile_size_adj = round(t_size * b_scale * 2)
    elif mag == 20:
        height_adj = round(height * b_scale * 2)
        width_adj = round(width * b_scale * 2)
        tile_size_adj = round(t_size * b_scale * 2)
    else:
        validate = False

    if validate:
        im_tile_predict = np.zeros((height_adj, width_adj), 'uint8')  #  Initialize an empty array for storing prediction results

        for col in tqdm(range(cols - 1), desc=f"Processing columns of {s_name}"):
            for row in range(rows - 1):
                tile = np.array(generator.get_tile(level, (col, row)))[:, :, :3]
                bn = np.sum(tile[:, :, 0] < 5) + np.sum(np.mean(tile, axis=2) > 245)
                if (np.std(tile[:, :, 0]) + np.std(tile[:, :, 1]) + np.std(tile[:, :, 2])) / 3 > 18 and bn < t_size * t_size * 0.3:
                    img_norm = reinhard(tile, reference_mu_lab, reference_std_lab, src_mu=s_mu_lab, src_sigma=s_sigma_lab)
                    img_tensor = transforms.ToTensor()(img_norm).unsqueeze(0).to(device)

                    with torch.no_grad():
                        m_model.eval()
                        logits = m_model(img_tensor)
                        pred_prob = torch.sigmoid(logits)

                    #print(f"Min: {pred_prob.min()}, Max: {pred_prob.max()}")  # Print min and max probability values

                    pred_prob = pred_prob.squeeze().cpu().numpy()
                    # Normalize and then scale for visualization
                    norm_mask = (pred_prob - np.min(pred_prob)) / (np.max(pred_prob) - np.min(pred_prob))
                    grayscale_mask = (norm_mask * 255).astype(np.uint8)
                    resized_mask = cv2.resize(grayscale_mask, (tile_size_adj, tile_size_adj))
                    im_tile_predict[row * tile_size_adj:row * tile_size_adj + tile_size_adj,
                                    col * tile_size_adj:col * tile_size_adj + tile_size_adj] = resized_mask


        output_path = os.path.join(out_path, s_name.split('.')[0] + '_til_mask_grayscale.png')
        plt.imsave(output_path, im_tile_predict, cmap='gray')

def main():
    if len(sys.argv) != 5:
        print("Usage: ", sys.argv[0], "<path to the WSI directory> <path to the output directory> <model path> <color normalization stats file>")
        exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_path = sys.argv[3]
    norm_stats = sys.argv[4]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.access(output_dir, os.W_OK):
        exit(1)

    tile_size = 256
    base_scale = 1 / 128.

    global reference_mu_lab, reference_std_lab
    reference_mu_lab = [8.63234435, -0.11501964, 0.03868433]
    reference_std_lab = [0.57506023, 0.10403329, 0.01364062]
    print("Loading model...")

    # Load the model architecture
    model = Unet(encoder_name='resnet50', encoder_weights='imagenet', classes=1, activation=None)

    # Check if multiple GPUs are available and use DataParallel if so
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Load the model state dict
    torch.load(model_path, map_location=device)
     

    # Move the model to the appropriate device
    model.to(device)
    model.eval()

    # Verify that weights are loaded correctly
    is_loaded = any(param.data.abs().sum() > 0 for param in model.parameters())
    if is_loaded:
        print("Model weights loaded successfully.")
    else:
        print("Warning: Model weights might not be loaded correctly.")

    df = pd.read_csv(norm_stats)
    whole_slide_images = sorted([f for f in os.listdir(input_dir) if f.endswith('.svs')])

    for img_name in tqdm(whole_slide_images, desc="Processing slides"):
        src_df = df.loc[df['slidename'] == img_name].to_numpy()[:, 1:].astype(np.float64)
        if len(src_df) != 0:
            src_mu_lab = src_df[0, :3]
            src_sigma_lab = src_df[0, 3:]
            slide_path = os.path.join(input_dir, img_name)
            try:
                slide_prediction(slide_path, model, src_mu_lab, src_sigma_lab, output_dir, img_name, tile_size, base_scale)
            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")
        else:
            print(f"No normalization data found for {img_name}")

if __name__ == "__main__":
    main()
