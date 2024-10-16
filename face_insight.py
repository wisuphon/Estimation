import cv2
import insightface
from insightface.app import FaceAnalysis
import onnxruntime as ort
import time

# กำหนดบริบทการทำงาน: -1 สำหรับ CPU, 0 หรือมากกว่า สำหรับ GPU
ctx_id = 0  # ใช้ -1 หากไม่มี GPU

# สร้างอินสแตนซ์ของ FaceAnalysis
# app = FaceAnalysis(providers=['CPUExecutionProvider'])  # หากมี GPU ให้เปลี่ยนเป็น ['CUDAExecutionProvider', 'CPUExecutionProvider']
# app.prepare(ctx_id=ctx_id)

session_options = ort.SessionOptions()
session_options.add_session_config_entry('gpu_mem_limit', '4294967296')
model_path = "models"
# model_pack_name = 'antelopev2'
# app = FaceAnalysis(name=model_pack_name, providers=['CPUExecutionProvider'], root=model_path)
app = FaceAnalysis(providers=['CUDAExecutionProvider'], root=model_path, session_options=session_options)
app.prepare(ctx_id=ctx_id, det_size=(640, 640))


def analyze(image_path):
    # โหลดภาพที่ต้องการวิเคราะห์
    # img_path = 'deepface/tests/dataset/img25.jpg'  # เปลี่ยนเป็นเส้นทางของภาพของคุณ
      # เปลี่ยนเป็นเส้นทางของภาพของคุณ
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image at {image_path}")
        return
    faces = app.get(img)

    for face in faces:

        age = face.age
        gender = 'Male' if face.gender == 1 else 'Female'
        # rang_age = ""
        # if (age <= 18):
        #     rang_age = "<18"
        # elif (age <= 30):
        #     rang_age = "19-30"
        # elif (age <= 40):
        #     rang_age = "31-40"
        # elif (age <= 50):
        #     rang_age = "41-50"
        # elif (age > 50):
        #     rang_age = "50+"

        print("Gender:" ,gender, "Rang age:", age)

    del faces

if __name__ == '__main__':
    # i = 0
    # while True:
    #     i += 1
        # print(i)
        time_run = time.time()
        # analyze('Image/t1.jpg')
        analyze('analyze.jpg')
        print("Total Time: ", time.time() - time_run)