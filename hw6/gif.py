from glob import glob
from PIL import Image
from argparse import ArgumentParser

parse = ArgumentParser()
parse.add_argument("path")
args = parse.parse_args()

files = glob(f"{args.path}/*.png")
files.sort()



# Take list of paths for images
image_path_list = ['dog-1.jpg', 'dog-2.jpg', 'dog-3.jpg']


# Create a list of image objects
image_list = [Image.open(file) for file in files]

# Save the first image as a GIF file
image_list[0].save(
            'animation.gif',
            save_all=True,
            append_images=image_list[1:], # append rest of the images
            duration=1000, # in milliseconds
            loop=0)