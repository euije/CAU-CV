import cv2
import numpy as np

# 영상 임포팅
img = cv2.imread('../assets/figures.png', 0)  # 이미지 경로를 'your_image_path.png'로 가정

# 이미지가 잘 불러왔는지 예외처리
if img is None:
    raise FileNotFoundError("이미지 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
else:
    print("이미지가 성공적으로 불러와졌습니다.")

# 물체 영역을 픽셀값 0으로 설정
_, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 거리 변환 수행
# Euclidean
dist_transform_euclidean = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
# City block (Manhattan)
dist_transform_cityblock = cv2.distanceTransform(binary_img, cv2.DIST_L1, 5)
# Chess board
dist_transform_chessboard = cv2.distanceTransform(binary_img, cv2.DIST_C, 5)


# 저장 최종 결과를 결과를 정규화하여 그레이 영상으로 저장
def save_normalized_image(dist_transform, filename):
    norm_img = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.uint8(norm_img)
    cv2.imwrite(filename, norm_img)


save_normalized_image(dist_transform_euclidean, 'dist_transform_euclidean.png')
save_normalized_image(dist_transform_cityblock, 'dist_transform_cityblock.png')
save_normalized_image(dist_transform_chessboard, 'dist_transform_chessboard.png')
