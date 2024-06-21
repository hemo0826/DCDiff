import os
import openslide
from PIL import Image


def generate_thumbnails(input_folder, output_folder, thumbnail_size=(128, 128)):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".svs") or filename.endswith(".tif"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"thumb_{os.path.splitext(filename)[0]}.png")

            # Open the pathology image
            slide = openslide.OpenSlide(input_path)

            # Generate the thumbnail
            thumbnail = slide.get_thumbnail(thumbnail_size)

            # Save the thumbnail
            thumbnail.save(output_path)


# Set the paths for the input and output folders
input_folder = 'path/to/input_folder'
output_folder = 'path/to/output_folder'

# Generate thumbnails
generate_thumbnails(input_folder, output_folder)
