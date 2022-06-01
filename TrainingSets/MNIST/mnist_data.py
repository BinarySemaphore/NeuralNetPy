'''
Source: http://yann.lecun.com/exdb/mnist/

Usisng data (in code):
import mnist_data as md
labels, images = md.load(start=0, end=images_per_set)

Drawing fun (in python shell):
>>> import mnist_data as md
>>> labels, images = md.load(start=0, end=100, flat=False)
>>> md.draw(images[0], cut_off=50)
or
>>> md.draw_set(images, labels, cut_off=50)
'''

import os
import struct


MAGIC_NUMBER_ID_LABELS = 2049
MAGIC_NUMBER_ID_IMAGES = 2051
DEFAULT_SET_GROUP = 'training'
FILE_SET_GROUPS = {
    'training': {
        'label': "train-labels.idx1-ubyte",
        'image': "train-images.idx3-ubyte"
    },
    'testing': {
        'label': "t10k-labels.idx1-ubyte",
        'image': "t10k-images.idx3-ubyte"
    }
}

LOAD_DIR = os.path.split(__file__)[0]


def load_labels(filename=FILE_SET_GROUPS[DEFAULT_SET_GROUP]['label']):
    count = 0
    labels = []
    
    full_filname = os.path.join(LOAD_DIR, filename)
    with open(full_filname, 'rb') as f:
        mgk_num = struct.unpack('>i', f.read(4))[0]
        if mgk_num != MAGIC_NUMBER_ID_LABELS:
            raise Exception("Aborted read of MNIST label data from invalid file \"%s\"" % filename)
        
        count = struct.unpack('>i', f.read(4))[0]
        for i in range(count):
            raw_val = f.read(1)
            val = int.from_bytes(raw_val, byteorder='big', signed=False)
            labels.append(val)
    
    if len(labels) != count:
        raise Exception("Failed read of MNIST label data from file \"%s\" due to count and label mismatch: expected %d count, read %d labels" % (filename, count, len(labels)))
    
    return labels


def load_images(filename=FILE_SET_GROUPS[DEFAULT_SET_GROUP]['image'], start=0, end=100, flat=True):
    count = 0
    images = []
    
    full_filname = os.path.join(LOAD_DIR, filename)
    with open(full_filname, 'rb') as f:
        mgk_num = struct.unpack('>i', f.read(4))[0]
        if mgk_num != MAGIC_NUMBER_ID_IMAGES:
            raise Exception("Aborted read of MNIST image data from invalid file \"%s\"" % filename)
        
        count = struct.unpack('>i', f.read(4))[0]
        n_rows = struct.unpack('>i', f.read(4))[0]
        n_columns = struct.unpack('>i', f.read(4))[0]
        seek_delta = n_rows * n_columns
        
        file_seek_to_start_of_images = f.tell()
        
        if end is None:
            end = count
        
        actual_range = end - start
        start = start % count
        f.seek(start * seek_delta, os.SEEK_CUR)
        
        for image_idx in range(start, start+actual_range):
            if image_idx == count:
                f.seek(file_seek_to_start_of_images, os.SEEK_SET)
            image = []
            for column in range(n_columns):
                image_column = []
                for row in range(n_rows):
                    raw_val = f.read(1)
                    val = int.from_bytes(raw_val, byteorder='big', signed=False)
                    image_column.append(val)
                if flat:
                    image.extend(image_column)
                else:
                    image.append(image_column)
            images.append(image)
    
    return images


def load(data_set=DEFAULT_SET_GROUP, start=0, end=100, flat=True):
    if end is not None and end <= start:
        raise Exception("end must be larger than start")
    
    label_filename = FILE_SET_GROUPS[data_set]['label']
    image_filename = FILE_SET_GROUPS[data_set]['image']
    
    labels = load_labels(label_filename)
    images = load_images(image_filename, start=start, end=end, flat=flat)
    
    if end is None:
        return labels[start:], images
    
    count = len(labels)
    actual_labels = []
    start = start % count
    end = end % count
    
    if start == 0 and end == 0:
        actual_labels = labels
    elif start > end:
        actual_labels.extend(labels[start:])
        actual_labels.extend(labels[:end])
    else:
        actual_labels = labels[start:end]
    
    #return list(map(list, zip(actual_labels, images)))
    return actual_labels, images


def draw(image, cut_off=0, isbitmap=False):
    if not isinstance(image[0], list):
        raise Exception("Expected non-flat image data; make sure load had 'flat=False'")
    
    ASCII_INTENSITY = [' ', '-', '~', '=', ')', '>', '<', '(', '}', '{', '+', '[', ']', '!', ':', '%', '#', '@', '7', '?', 'Z', 'C', 'L', 'f', 'V', 'i', 'J', 'X', 'Y', 't', 'T', 'l', 'F', 'A', '3', 'I', '2', '5', 'S', 'K', 'M', 'E', '1', '9', '6', '4', 'b', 'P', 'O', 'G', 'U', 'R', '$', '0', '8', '&', 'H', 'N', 'W', 'D', 'B', 'Q']
    
    if cut_off:
        ASCII_INTENSITY = ASCII_INTENSITY[1:]
    intensity_div = (255.0 - cut_off) / (len(ASCII_INTENSITY) - 1)
    
    output = []
    for column in image:
        for val in column:
            if isbitmap:
                val *= 255
            # Append twice to double pixel width
            if cut_off and val <= cut_off:
                output.append(' ' * 2)
            else:
                if cut_off:
                    val = val - cut_off
                intensity_index = int(val / intensity_div)
                output.append(ASCII_INTENSITY[intensity_index] * 2)
        output.append('|\n')
    
    print("".join(output))


def draw_set(images, labels=[], cut_off=0):
    has_labels = len(labels) > 0
    for index in range(len(images)):
        image = images[index]
        print('=' * 50)
        if has_labels:
            label = labels[index]
            print('\t\t%s' % label)
            print('-' * 50)
        draw(image, cut_off=cut_off)


def draw_flat(flat_image, width=28, cut_off=0, isbitmap=False):
    image = []
    image_column = []
    for index in range(len(flat_image)):
        image_column.append(flat_image[index])
        if len(image_column) == width:
            image.append(image_column)
            image_column = []
    draw(image, cut_off=cut_off, isbitmap=isbitmap)


def shrink(image, scale=2, width=28, flat=True):
    new_image = []
    if not flat:
        width = len(image[0])
    new_size = int(width / scale)
    for y in range(new_size):
        new_column = []
        for x in range(new_size):
            c_range = range(y * scale, (y + 1) * scale)
            r_range = range(x * scale, (x + 1) * scale)
            pixel_set = []
            for c_idx in c_range:
                for r_idx in r_range:
                    if flat:
                        val = image[r_idx + c_idx * 28]
                    else:
                        val = image[c_idx][r_idx]
                    pixel_set.append(val)
            new_val = int(sum(pixel_set) / len(pixel_set))
            new_column.append(new_val)
        if flat:
            new_image.extend(new_column)
        else:
            new_image.append(new_column)
    return new_image


def align_flat(image, left=0.0, right=0.0, top=0.0, bottom=0.0, relative=True, width=28, height=28):
    right_margin = 0
    left_margin = width
    top_margin = height
    bottom_margin = 0
    for y in range(height):
        row_has_something = False
        for x in range(width):
            if image[y*height+x] != 0:
                row_has_something = True
                if x > right_margin:
                    right_margin = x
                if x < left_margin:
                    left_margin = x
                if y < top_margin:
                    top_margin = y
        if row_has_something:
            if y > bottom_margin:
                bottom_margin = y
    right_margin = width - right_margin - 1
    bottom_margin = height - bottom_margin
    
    if left:
        left_move = left
        if relative:
            left_move = int(left_margin * left)
        left_padding = []
        for y in range(height):
            y_pos = y * width
            for x in range(left_move):
                left_padding.append(image[y_pos+x])
        for y in range(height):
            y_pos_pop = y * width
            y_pos_ins = (y + 1) * width
            y_pos_pad = y * left_move
            for x in range(left_move):
                image.pop(y_pos_pop)
                image.insert(y_pos_ins, left_padding[y_pos_pad+x])
    elif right:
        right_move = right
        if relative:
            right_move = int(right_margin * right)
        right_padding = []
        for y in range(height):
            y_pos = (y + 1) * width - 1
            for x in range(right_move):
                right_padding.append(image[y_pos-x])
        for y in range(height):
            y_pos_pop = (y + 1) * width - right_move
            y_pos_ins = y * width
            y_pos_pad = y * right_move
            for x in range(right_move):
                image.pop(y_pos_pop)
            for x in range(right_move):
                image.insert(y_pos_ins, right_padding[y_pos_pad+x])
    
    if top:
        top_move = top
        if relative:
            top_move = int(top_margin * top)
        top_padding = []
        for y in range(top_move):
            y_pos = y * width
            for x in range(width):
                top_padding.append(image[y_pos+x])
        for y in range(top_move):
            y_pos_pad = y * top_move
            for x in range(width):
                image.pop(0)
                image.append(top_padding[y_pos_pad+x])
    elif bottom:
        bottom_move = bottom
        if relative:
            bottom_move = int(bottom_margin * bottom)
        bottom_padding = []
        for y in range(bottom_move):
            y_pos = (height - y - 1) * width
            for x in range(width):
                bottom_padding.append(image[y_pos+x])
        for y in range(bottom_move):
            y_pos_pad = (y + 1) * width - 1
            for x in range(width):
                image.pop(-1)
            for x in range(width):
                image.insert(0, bottom_padding[y_pos_pad-x])
