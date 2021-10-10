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


def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    gem = cv.imread('Diamond.png', cv.IMREAD_UNCHANGED)
    gem = img_resize(gem, height = 64)

    ret, new_frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit(0)

    old_frame = np.zeros(new_frame.shape, dtype=np.uint8)
    dest_frame = np.zeros(new_frame.shape, dtype=np.uint8)
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

        final_frame = overlay_transparent(new_frame, gem, 100, 100)

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