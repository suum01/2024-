from rgbmatrix import RGBMatrix, RGBMatrixOptions
from PIL import Image
import time

# LED 매트릭스 옵션 설정
options = RGBMatrixOptions()
options.rows = 64   # LED 패널의 행 수
options.cols = 64   # LED 패널의 열 수
options.chain_length = 1  # 체인에 연결된 패널의 수
matrix = RGBMatrix(options=options)

# GIF 파일을 열기
image = Image.open("/mnt/data/dino.gif")

# GIF 파일의 각 프레임을 LED 매트릭스에 표시
while True:
    for frame in range(image.n_frames):
        image.seek(frame)
        frame_image = image.resize((64, 64))  # 64x64로 크기 조정
        frame_image = frame_image.convert('RGB')  # RGB 형식으로 변환
        matrix.SetImage(frame_image)
        time.sleep(0.1)  # 각 프레임을 0.1초 동안 표시
