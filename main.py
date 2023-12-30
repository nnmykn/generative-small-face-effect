import dlib
import cv2
import numpy as np
from PIL import Image

# 画像ファイルを読み込む
image_path = './IMG_7361.JPG'
cv_image = cv2.imread(image_path)

# グレースケールに変換
gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

predictor_path = "./shape_predictor_68_face_landmarks.dat"
shrink_factor = 0.9  # 顔を小さくする係数

try:
    predictor = dlib.shape_predictor(predictor_path)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)

    seamless = cv_image.copy()  # デフォルトの結果として元の画像を使用

    for face in faces:
        landmarks = predictor(gray, face)
        points = np.array([[p.x, p.y] for p in landmarks.parts()])

        # mask の初期化（3チャンネル化）
        mask = np.zeros_like(cv_image, dtype=np.uint8)

        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, (255, 255, 255))

        rect = cv2.boundingRect(hull)
        if (rect[2] > 0 and rect[3] > 0 and
                rect[0] + rect[2] <= cv_image.shape[1] and
                rect[1] + rect[3] <= cv_image.shape[0]):
            subdiv = cv2.Subdiv2D(rect)
            for p in points:
                subdiv.insert((int(p[0]), int(p[1])))
            triangles = subdiv.getTriangleList()
            triangles = np.array(triangles, dtype=np.int32)

            for t in triangles:
                pts1 = np.float32([t[0:2], t[2:4], t[4:6]])
                if all((0 <= x < cv_image.shape[1] and 0 <= y < cv_image.shape[0]) for x, y in pts1):
                    center = np.mean(pts1, axis=0)
                    pts2 = np.float32([center + (v - center) * shrink_factor for v in pts1])
                    warp_matrix = cv2.getAffineTransform(pts1, pts2)
                    warped_image = cv2.warpAffine(cv_image, warp_matrix,
                                                  (cv_image.shape[1], cv_image.shape[0]),
                                                  borderMode=cv2.BORDER_REFLECT)
                    # マスクを使用して変形した画像をオーバーレイ
                    seamless = np.where(mask == 255, warped_image, seamless)

    result = cv2.cvtColor(seamless, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result)
    result_img_path = './seamless_IMG_7361.JPG'
    result_pil.save(result_img_path)

except Exception as e:
    result_img_path = f"Error: {str(e)}"

print(result_img_path)
