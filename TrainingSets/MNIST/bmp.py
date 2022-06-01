# BMP to MNIST
import os
import struct


def load(filename, invert=True, flat=True):
    image = []
    with open(filename, 'rb') as f:
        header_type = f.read(2)
        if header_type != b'BM':
            raise Exception("Cannot load \"%s\" as a bitmap; unexpected header '%r'" % (filename, header_type))
        
        f.seek(18)
        width, height = struct.unpack('ii', f.read(8))
        f.seek(28)
        bits_per_pixel = struct.unpack('H', f.read(2))[0]
        
        if bits_per_pixel not in (8, 24):
            raise Exception("Expected 8-bit or 24-bit BMP file: given %d-bit file" % bits_per_pixel)
        
        f.seek(54)
        for y in range(height):
            image_row = []
            for x in range(width):
                rgb = []
                for i in range(3):
                    raw_val = f.read(1)
                    val = int.from_bytes(raw_val, byteorder='big', signed=False)
                    rgb.append(val)
                intensity = int(sum(rgb) / 3)
                if invert:
                    intensity = 255 - intensity
                image_row.append(intensity)
            image.append(image_row)
        image.reverse()
        if flat:
            # Need to flatten in post-processing because rows are read bottom-to-top
            flat_image = []
            for image_row in image:
                flat_image.extend(image_row)
            image = flat_image
    
    return image
