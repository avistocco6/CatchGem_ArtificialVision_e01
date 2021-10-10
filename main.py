from time import sleep

import numpy as np
import cv2 as cv

def img_resize(image, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def insert_img(background, img, x_offset, y_offset):
    """Inserts a small image in a background at a random position"""
    x_end = x_offset + img.shape[1]
    y_end = y_offset + img.shape[0]
    background[y_offset:y_end, x_offset:x_end] = img

    return background

def bland_img(background, img, x_offset, y_offset):
    rows, columns, channels = img.shape
    roi = background[x_offset:(x_offset+rows), y_offset:(y_offset+columns)]

    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    ret, mask = cv.threshold(img_gray, 120, 255, cv.THRESH_BINARY)

    bg = cv.bitwise_or(roi, roi, mask=mask)
    mask_inv = cv.bitwise_not(img_gray)
    fg = cv.bitwise_and(img, img, mask=mask_inv)

    final_roi = cv.add(bg, fg)
    cv.imshow("fg", final_roi)
    img = final_roi
    background[y_offset : y_offset + img.shape[0], x_offset : x_offset + img.shape[1]]= img

    return background

def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    gem = cv.imread('Diamond.png', cv.IMREAD_UNCHANGED)
    # gem = cv.resize(gem,(64,64))
    gem = img_resize(gem, height = 64)
    if len(gem.shape) > 2 and gem.shape[2] == 4:
        # convert the image from RGBA2RGB
        gem = cv.cvtColor(gem, cv.COLOR_BGRA2BGR)

    old_frame = np.zeros((480,640,3), dtype=np.uint8)
    dest_frame = np.zeros((480,640,3), dtype=np.uint8)
    while True:
        # Capture frame-by-frame
        ret, new_frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Our operations on the frame come here
        cv.subtract(new_frame, old_frame, dest_frame)
        ret, thresh1 = cv.threshold(dest_frame, 100, 255, cv.THRESH_BINARY)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # final_frame = insert_img(new_frame, gem)
        final_frame = insert_img(new_frame, gem, 100, 100)

        # Display the resulting frame
        flipHorizontal = cv.flip(final_frame, 1)
        cv.imshow('frame', flipHorizontal)
        old_frame = new_frame

        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

main()