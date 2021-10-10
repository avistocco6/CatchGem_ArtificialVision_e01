import numpy as np
import cv2 as cv
import time
import random

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
    gem = img_resize(gem, height=64)
    gem_h, gem_w = gem.shape[0], gem.shape[1]

    bomb = cv.imread('Bomb.png', cv.IMREAD_UNCHANGED)
    bomb = img_resize(bomb, height=64)
    bomb_h, bomb_w = bomb.shape[0], bomb.shape[1]

    ret, new_frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit(0)

    score = 0
    delay = 2.5
    time_on_screen = 3

    old_frame = np.zeros(new_frame.shape, dtype=np.uint8)
    dest_frame = np.zeros(new_frame.shape, dtype=np.uint8)

    current_objects = list()

    start_time = time.time()
    while True:
        # Capture frame-by-frame
        ret, new_frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Our operations on the frame come here
        cv.subtract(new_frame, old_frame, dest_frame)
        ret, thresh1 = cv.threshold(dest_frame, 200, 255, cv.THRESH_BINARY)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        elapsed_time = time.time() - start_time
        if elapsed_time >= delay:
            start_time = time.time()
            if random.randint(1, 3) == 1:
                x = random.randint(20, new_frame.shape[0]-bomb_w-20)
                y = random.randint(20, new_frame.shape[1]/2-bomb_h-20)
                current_objects.append(["bomb", x, y, time_on_screen])
            else:
                x = random.randint(20, new_frame.shape[0]-gem_w-20)
                y = random.randint(20, new_frame.shape[1]/2-gem_h-20)
                current_objects.append(["gem", x, y, time_on_screen])

        for data in current_objects:
            x = data[1]
            y = data[2]
            if elapsed_time >= 1:
                data[3] -= 1

            if data[0] == "bomb":
                if np.any(thresh1[y:y + bomb_h, x:x + bomb_w]):
                    if score > 0:
                        score -= 1
                    print("OPS, BOMB TAKEN! Current score: ", score)
                    current_objects.remove(data)
                elif data[3] == 0:
                    current_objects.remove(data)
                else:
                    new_frame = overlay_transparent(new_frame, bomb, x, y)
            else:
                if np.any(thresh1[y:y + gem_h, x:x + gem_w]):
                    score += 1
                    current_objects.remove(data)
                    print("GEM TAKEN! Current score: ", score)
                elif data[3] == 0:
                    current_objects.remove(data)
                else:
                    new_frame = overlay_transparent(new_frame, gem, x, y)

        # Display the resulting frame
        flip_horizontal = cv.flip(new_frame, 1)
        flip_horizontal_motion = cv.flip(dest_frame, 1)
        cv.imshow('frame', flip_horizontal)
        cv.imshow('motion', flip_horizontal_motion)
        old_frame = new_frame

        if cv.waitKey(40) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

main()