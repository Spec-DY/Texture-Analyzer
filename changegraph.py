import os
from PIL import Image, ImageOps
from tqdm import tqdm


input_folder = "./textures/wood"
output_folder = "./newtexture/wood_mirror"
os.makedirs(output_folder, exist_ok=True)

for filename in tqdm(os.listdir(input_folder), desc="processing image"):

    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):

        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # mirror the original image
        mirror_img = ImageOps.mirror(img)
        # add _mirror to the filename
        base_name, ext = os.path.splitext(filename)
        new_filename = f"{base_name}_mirror{ext}"

        mirror_img.save(os.path.join(output_folder, new_filename))
