from PIL import Image
import numpy as np
import cv2

# Load the image file
img_path = '/mnt/data/IMG_7361.JPG'
original_image = Image.open(img_path)

# Convert to an array
image_array = np.array(original_image)

# Use OpenCV to load the image for face detection
cv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

# Load the default face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# For the sake of this example, we'll only focus on the first detected face
for (x, y, w, h) in faces:
    # Let's make the face 10% smaller
    shrink_factor = 0.9
    new_w = int(w * shrink_factor)
    new_h = int(h * shrink_factor)

    # Calculate the offset needed to center the smaller face
    offset_w = int((w - new_w) / 2)
    offset_h = int((h - new_h) / 2)

    # Create a new image with a smaller face
    face_img = cv_image[y + offset_h:y + h - offset_h, x + offset_w:x + w - offset_w]
    face_img = cv2.resize(face_img, (w, h))

    # Put the resized face back into the original image
    cv_image[y:y+h, x:x+w] = face_img

# Convert back to RGB and save the edited image
edited_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
edited_pil_image = Image.fromarray(edited_image)
edited_img_path = '/mnt/data/edited_IMG_7361.JPG'
edited_pil_image.save(edited_img_path)

edited_img_path
