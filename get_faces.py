import os
import cv2
import sys

def detect_faces(cascade, test_image):
    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces_rect:
        #cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)    # <-- for drawing rectangle around face
        # crop image around face
        test_image = test_image[y:y + h, x:x + w]
        # resize
        test_image = cv2.resize(test_image,(256,256),interpolation=cv2.INTER_AREA)
        
        return test_image
    
    

def faces_from_video(video_path):
    outdir = os.path.splitext(os.path.basename(video_path))[0]+'_images'
    os.mkdir(outdir)
    print('Creating output folder with name "{}".'.format(outdir))
    skipdir = outdir + "_SKIPPED"

    haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Opens the Video file
    cap= cv2.VideoCapture(video_path)
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        #call the face detection function
        face = detect_faces(haar_face_cascade, frame)
        try:
            cv2.imwrite(os.path.join(outdir, 'face'+str(i).zfill(8)+'.jpg'), face)
        except:
            print("no face detected - Skipping.")
            try:
                os.mkdir(skipdir)
            except:
                pass
            cv2.imwrite(os.path.join(skipdir, 'skipped_image'+str(i).zfill(8)+'.jpg'), frame)

        i+=1

    cap.release()
    cv2.destroyAllWindows()
    
    
    
def faces_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):
            path = folder_path + "/" + filename
            faces_from_video(path)
            continue
        else:
            continue
    
    
    
if __name__ == '__main__':
    path = sys.argv[1]
    if os.path.isdir(path):
        faces_from_folder(path)
    else:
        faces_from_video(path)