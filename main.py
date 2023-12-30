import dlib
import cv2

# Load the image in grayscale
gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

# Initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor using the path to the dat file
# which is assumed to be in the working directory.
predictor_path = "./shape_predictor_68_face_landmarks.dat"

# The facial landmark predictor is required to be in the same directory as this script.
# As internet access is not available, we will attempt to use the file if it is present,
# but this step is expected to fail since we cannot download the predictor data file.
try:
    # Create the facial landmark predictor
    predictor = dlib.shape_predictor(predictor_path)

    # Now that we have the detector and predictor, we can process the image:
    for (x, y, w, h) in faces:
        # Create a dlib rectangle object from the bounding box coordinates
        rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)

        # Use the predictor to find the landmark points
        landmarks = predictor(gray, rect)

        # Convert landmarks to numpy array
        points = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Define a mask for the face which is the same size as the original image
        mask = np.zeros_like(cv_image, dtype=np.uint8)

        # We will use convex hull to create a polygon around the landmarks
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, (255, 255, 255))

        # Create a Delaunay triangulation of the points
        rect = cv2.boundingRect(hull)
        subdiv = cv2.Subdiv2D(rect)
        for p in points:
            subdiv.insert((int(p[0]), int(p[1])))
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        # For each triangle, apply a local warp to make the face smaller
        for t in triangles:
            pts1 = np.float32([t[0:2], t[2:4], t[4:6]])
            # Compute the centroid of the triangle
            center = np.mean(pts1, axis=0)
            # Move the vertices of the triangle towards the centroid to make the face smaller
            pts2 = np.float32([center + (v - center) * shrink_factor for v in pts1])
            # Apply the warp to a small region within the triangle
            cv2.warpAffine(cv_image, cv2.getAffineTransform(pts1, pts2),
                           (cv_image.shape[1], cv_image.shape[0]),
                           dst=mask,
                           borderMode=cv2.BORDER_REFLECT,
                           flags=cv2.WARP_INVERSE_MAP)

        # Blend the masked small face with the original image
        seamless = cv2.seamlessClone(mask, cv_image, mask, (x+w//2, y+h//2), cv2.NORMAL_CLONE)

    # Save the result
    result = cv2.cvtColor(seamless, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result)
    result_img_path = '/mnt/data/seamless_IMG_7361.JPG'
    result_pil.save(result_img_path)

except Exception as e:
    # If any error occurs, we'll just return a message
    result_img_path = f"Error: {str(e)}"

result_img_path
