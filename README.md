# equirements
- mediapipe 0.8.4
- OpenCV 4.6.0.66 or Later
- Tensorflow 2.9.0 or Later
- protobuf <3.20,>=3.9.2
- scikit-learn 1.0.2 or Later
- matplotlib 3.5.1 or Later

# app.py 실행 시
1. 미리 학습된 손모양 데이터 모델로 불러옴
2. 연결된 웹캠 감지
3. mediapipe로 손 객체 감지
4. 손모양이 16프레임동안 포인팅하는 모양이라면 검지손가락 끝을 기준으로 위쪽이미지를 남기고 크롭됨
    - 크롭된 이미지(cropped_image)가 cv2.imshow로 표시되고 ./images/output.png 파일로 저장됨
    - 주석으로 '포인팅 홀드 감지' 라고 표시된 부분 참조할 것
5. 16프레임동안 포인팅하는 모양이 아니라면 cv2.imshow로 표시된 창이 종료되고, 다시 포인팅 홀드 감지될 때까지 대기
