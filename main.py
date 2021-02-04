import cv2
#opencv documentation
#https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html?highlight=detectmultiscale

video = cv2.VideoCapture("Car_Clip.mp4") #Name of the clip 
Car_Classifier = "cars.xml" #Car classifier xml file-this xml will check our image and check it against the data and if it passed then it will be classed as a car
CheckCar = cv2.CascadeClassifier(Car_Classifier)

# Reference https://gist.github.com/199995/37e1e0af2bf8965e8058a9dfa3285bc6#file-cars-xml

while True:#As we are reading frames from a video it is better to use a while loop so it can go through each frame until the video ends
    (read_status, frame) = video.read()#This will capture the current frame from the video and call it frame and the read status returns if the read was successful
    if read_status:
        Grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break#if the read was unsuccessful break out of the while loop this will happen at the end of the video

    CarTracker = CheckCar.detectMultiScale(Grayscale)#this will apply the car classifier to the image and then will give out  coordinates for each car found
#if i were to print this then it would give the coordinates of the car
    for(X, Y,W, H) in CarTracker:#This will take the coordinates given in car tracker and then from that it will draw a rectangle to so the cars position
        cv2.rectangle(frame, (X, Y), (X+W, Y+H), (255, 0, 0), 2)  #This is drawing the rectangle from the given coordinates from each frame

    cv2.imshow('Car detection',frame)#opens the file that we just made and calls the window "Car detection"
    cv2.waitKey(1)#stops the window from autoclosing-closes when its the end of the video
