import cv2
import mediapipe as mp
import time
from bs4 import BeautifulSoup
import requests
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

class HandDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.driver = webdriver.Chrome()
    
    def detect_hands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, ":", cx, cy)
                    if id == 4 or id == 8:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

                self.mp_drawing.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)

                if abs(handLms.landmark[4].x * w - handLms.landmark[8].x * w) <= 9 and abs(handLms.landmark[4].y * h - handLms.landmark[8].y * h) <= 9:
                    print("OK 싸인!!!")
                    command = "검색어"
                    self.site(command)

        cv2.imshow("Gotcha", img)

    def site(self, command):
        url = 'https://www.naver.com/'
        response = requests.get(url)

        if response.status_code == 200:
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
        else:
            print("사이트 접속 실패", response.status_code)

        ## ok 싸인 받고 창이 꺼지는 현상을 방지하는 코드 추가
        chrome_options = Options()
        chrome_options.add_experimental_option('detach', True)
        chrome_options.add_argument('--start-maximized')


        driver = webdriver.Chrome(options=chrome_options)
        
        
        self.driver.get(url)
        time.sleep(1)

        search = self.driver.find_element(By.ID, 'query')
        search.send_keys(command)
        search.send_keys(Keys.RETURN)

    def run(self):
        while True:
            success, img = self.cap.read()
            if not success:
                continue

            self.detect_hands(img)

            key = cv2.waitKey(1)
            if key == ord('c'):
                self.driver.close()  # 현재 열린 창만 닫음
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    hand_detector = HandDetector()
    hand_detector.run()
