import cv2
import numpy as np

proto = 'models/colorization_deploy_V2.prototxt'
weights = 'models/colorization_release_v2.caffemodel'

net = cv2.dnn.readNetFromCaffe(proto, weights)

pts_in_hull = np.load('models/pts_in_hull.npy')
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]

net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full((1, 313), 2.606, np.float32)]


def colorization(img, mask=False):
    h, w, c = img.shape

    img_input = img.copy()

    # uint8 -> float32
    img_input = img_input.astype("float32") / 255.
    # ColorSystem : BGR -> Lab
    img_lab = cv2.cvtColor(img_input, cv2.COLOR_BGR2Lab)
    # Get L channel
    img_l = img_lab[:, :, 0:1]

    # Preprocessing function
    blob = cv2.dnn.blobFromImage(img_l, size = (224, 224), mean = [50, 50, 50])

    # inference
    net.setInput(blob)
    output = net.forward()

    # Post processing
    output = output.squeeze().transpose((1, 2, 0))

    # Restore image size
    output_resized = cv2.resize(output, (w, h))

    output_lab = np.concatenate([img_l, output_resized], axis = 2)

    # ColorSystem : Lab -> BGR
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_Lab2BGR)
    output_bgr = output_bgr * 255
    output_bgr = np.clip(output_bgr, 0, 255)
    output_bgr = output_bgr.astype('uint8')

    return output_bgr