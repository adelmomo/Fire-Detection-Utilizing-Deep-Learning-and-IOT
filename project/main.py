import cv2
import pygame
import tensorflow as tf
import numpy as np
from model import *
import threading
from picamera.array import PiRGBArray
from picamera import PiCamera
pygame.init()
pos,neg=0,0
class PiVideoStream:
    def __init__(self, resolution=(320, 240), framerate=32):
            # initialize the camera and stream
            self.camera=PiCamera()
            self.camera.resolution=resolution
            self.camera.framerate = framerate
            self.rawCapture = PiRGBArray(self.camera, size=resolution)
            self.stream = self.camera.capture_continuous(self.rawCapture,
			format="bgr", use_video_port=True)
            # initialize the frame and the variable used to indicate
            # if the thread should be stopped
            self.frame = None
            self.stopped = False
    def start(self):
            # start the thread to read frames from the video stream
            th1=threading.Thread(target=self.update, args=())
            th1.start()
            
            
            return self
    def update(self):
            # keep looping infinitely until the thread is stopped
            for f in self.stream:
			# grab the frame from the stream and clear the stream in
			# preparation for the next frame
                    self.frame = f.array
                    self.rawCapture.truncate(0)
                    self.show()
			# if the thread indicator variable is set, stop the thread
			# and resource camera resources
                    if self.stopped:
                    
                        self.stream.close()
                        self.rawCapture.close()
                        self.camera.close()
                        return
    def read(self):
        if(self.frame is not None):
    # return the frame most recently read
            return self.frame
        else:
            return None

    def stop(self):
            # indicate that the thread should be stopped
            self.stopped = True
    def show(self):
        
        if(self.frame is not None):
                
                
            cv2.waitKey(1)
            cv2.imshow('',self.frame)
                
        
def play():
    try:
        pygame.mixer.music.load('fire-alarm.wav')
        pygame.mixer.music.play()
    except:
        print('')
    
fr=0
pos,neg=0,0
c=0
vs = PiVideoStream().start()
while 1:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels

    if(fr>=20):
        fr=0
        if(pos>neg):
            threading.Thread(target=play).start()
        pos,neg=0,0
    frame = vs.read()
    if(frame is not None):
        frame=cv2.resize(frame,(100,100))
                
        frame=(frame-mean)/std
        
        res=model(tf.expand_dims(frame,axis=0)).numpy()
        fr+=1
        if(res.argmax()==1):
          print('Fire')
          pos+=1
        else:
          print('Normal')
          neg+=1
	# check to see if the frame should be displayed to our screen
    

##while(1):
##    if(fr>=30):
##        fr=0
##        if(pos>neg):
##            threading.Thread(target=play).start()
##        pos,neg=0,0
##    frame = vs.read()
##    if(frame is not None):
##       # cv2.waitKey(1)
##       # cv2.imshow('',frame)
##        frame=cv2.resize(frame,(n,m))
##        frame=(frame-mean)/std
##        res=model.predict([[frame]])
##        if(res.argmax()==1):
##            pos+=1
##        else:
##            neg+=1
##
##    fr+=1


    
