# Evaluation pipeline

from PIL import Image, ImageDraw, ImageFont
import argparse
import math

#
# Mark point on image and compare to target
#

# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument("-name", required=True)
parser.add_argument("-x", type=int, required=True)
parser.add_argument("-y", type=int, required=True)
args = parser.parse_args()

image = Image.open("reference/carola68-stud-4999947.jpg") # Open the image
pixels = image.load() # Load the pixels into a matrix

# Set target coordinates manually
# See 'ground-truth.py' to generate these coordinates
x_target = 2644
y_target = 1320

# Highlight target center, add a marker cross for visibility
for i in range(20):
    pixels[x_target + i,y_target] = (0,255,0)
    pixels[x_target - i,y_target] = (0,255,0)
    pixels[x_target,y_target + i] = (0,255,0)
    pixels[x_target,y_target - i] = (0,255,0)

# Highlight vlm-pointed pixel, add a marker cross for visibility
for i in range(20):
    pixels[args.x + i,args.y] = (255,255,0)
    pixels[args.x - i,args.y] = (255,255,0)
    pixels[args.x,args.y + i] = (255,255,0)
    pixels[args.x,args.y - i] = (255,255,0)

# Calculate error ("distance") in pixels
error = round(math.dist((args.x, args.y), (x_target, y_target)))

# Add markings for visualization
drawing=ImageDraw.Draw(image)
font1 = ImageFont.load_default(size=30)

drawing.text((x_target + 10, y_target - 50), "target", fill =(0,255,0), font=font1)
drawing.text((args.x - 110, args.y), f"{args.name}", fill =(255, 255, 0), font=font1)

# Print results and save image
print("Distance from target:", error, "px")
image.save(f"{args.name}-marked.jpg")