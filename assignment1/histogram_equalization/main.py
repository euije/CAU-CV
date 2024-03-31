import cv2
import matplotlib.pyplot as plt

# Lena 이미지를 그레이스케일로 불러오기
lena_img = cv2.imread('../assets/lena.png', cv2.IMREAD_GRAYSCALE)

# 이미지가 잘 불러왔는지 예외처리
if lena_img is None:
    raise FileNotFoundError("Lena 이미지 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
else:
    print("Lena 이미지가 성공적으로 불러와졌습니다.")

# Lena 영상의 히스토그램 이퀄라이제이션 수행 결과 영상
equ = cv2.equalizeHist(lena_img)  # 히스토그램 이퀄라이제이션 적용

# 그래프와 이미지를 한 개의 plot에 표시
plt.figure(figsize=(16, 8))

# 원본 그레이스케일 이미지
plt.subplot(2, 3, 1)
plt.imshow(lena_img, cmap='gray')
plt.title('Original Grayscale Lena Image')
plt.axis('off')  # 축 표시 제거

# 원본 이미지의 히스토그램
plt.subplot(2, 3, 2)
plt.hist(lena_img.ravel(), 256, [0, 256])
plt.title('Histogram for Grayscale Lena Image')

# 원본 이미지의 누적 히스토그램
plt.subplot(2, 3, 3)
plt.hist(lena_img.ravel(), 256, [0, 256], cumulative=True)
plt.title('Cumulative Histogram')

# 이퀄라이제이션된 이미지
plt.subplot(2, 3, 4)
plt.imshow(equ, cmap='gray')
plt.title('Equalized Lena Image')
plt.axis('off')  # 축 표시 제거

# 이퀄라이제이션된 이미지의 히스토그램
plt.subplot(2, 3, 5)
plt.hist(equ.ravel(), 256, [0, 256])
plt.title('Histogram for Equalized Lena Image')

# 이퀄라이제이션된 이미지의 누적 히스토그램
plt.subplot(2, 3, 6)
plt.hist(equ.ravel(), 256, [0, 256], cumulative=True)
plt.title('Cumulative Histogram for Equalized Image')

plt.tight_layout()  # subplot 간격 자동 조정
plt.show()
