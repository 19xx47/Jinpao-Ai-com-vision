import cv2
import numpy as np

name = '/home/worakan/template/images-crop/front-crop.png' 
template = cv2.imread(name, 0)
face_w, face_h = template.shape[::-1]

cv2.namedWindow('image')

cap = cv2.VideoCapture('/home/worakan/template/3point.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
output_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for the output video
output_video = cv2.VideoWriter('output_video_front_rgb.mp4', fourcc, fps, output_size)
# print("Video Frame Rate:", fps)

threshold = 0.68
thresholdxmaxx = 0.69

ret = True

while ret :
    ret, img = cap.read()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)

    if len(res):
        location = np.where(np.logical_and(threshold <= res, res< thresholdxmaxx))
        for pt in zip(*location[::-1]):
            #puting  rectangle on recognized erea 
            cv2.rectangle(img, pt, (pt[0] + face_w, pt[1] + face_h), (0,0,255), 2)
            cv2.putText(img, f'conf:{res[pt[1], pt[0]]:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            print(f'conf:{res[pt[1], pt[0]]:.2f}')
    cv2.imshow('image',img)
    output_video.write(img)
    k = cv2.waitKey(103) & 0xFF
    if k == 27:
        break
cap.release()
output_video.release()
cv2.destroyAllWindows()
