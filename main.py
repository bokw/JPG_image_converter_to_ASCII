import cv2
import numpy as np

def scale(img: list) -> list:
    '''Scaling image down if width exceeds 100 pixels.'''
    img_dim = img.shape
    scale_val = 0.9
    if img_dim[1] > 100:
        # OpenCV uses Numpy. Numpy arrays use (height, width) while OpenCV uses (width, height). That's the reason for flip.
        w = int((img_dim[1])*scale_val)
        h = int((img_dim[0])*scale_val)
        scaled_img_dim = (w, h)
        scaled_img = cv2.resize(img, scaled_img_dim, interpolation=cv2.INTER_AREA)
        scaled_img = scale(scaled_img)
        return scaled_img
    else:
        return img


def greyscale(img: list) -> list:
    '''Calculating relative luminance. Based on: https://en.wikipedia.org/wiki/Relative_luminance'''
    grey_img = []
    img = scale(img)
    for row in img:
        luminance = []
        for val in row:
            lum = int(0.2126*val[0] + 0.7152*val[1] + 0.0722*val[2])
            luminance.append(lum)
        grey_img.append(luminance) 
    grey_img = np.array(grey_img)   
    return grey_img


def to_ascii(img: list) -> list:
    '''Image conversion to ASCII'''
    ascii_char = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
    max_pixel = 255
    ascii_matrix = []
    img = greyscale(img)
    for row in img:
        ascii_row = []
        for val in row:
            ascii_row.append((ascii_char[int(val/max_pixel * len(ascii_char))-1]))
        ascii_matrix.append(ascii_row)
    return ascii_matrix


def print_to_terminal(img: list):
    '''Prints ASCII image to terminal.'''
    print("\n")
    for row in to_ascii(img):
        print(''.join(row))
    print("\n")


def write_to_file(img: list):
    f = open("output.txt", "w")
    for row in to_ascii(img):
        f.write(''.join(row) + "\n")
    f.close()


def convert(img_pth):
    '''Outputs converted image to terminal and to file.'''
    image = cv2.imread(img_pth, 1) # 0 will render image grayscale, 1 in RGB - 1 is left for educational purposes
    print("Height and width of the original image:", image.shape[0], "x", image.shape[1])
    to_ascii(image)
    print("Height and width of the rescaled image:", greyscale(image).shape[0], "x", greyscale(image).shape[1])
    print("Success! Image converted to ASCII.")
    print_to_terminal(image)
    write_to_file(image)


image_path = r'C:\...\...\...\deer.jpg'
convert(image_path)
