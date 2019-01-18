import tensorflow as tf
import sys
import os
import cv2
import math


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

video_path = sys.argv[1]



vidoe_data = tf.gfile.FastGFile(video_path, 'rb').read()


label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("tf_files/retrained_labels.txt")]

with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
 
    graph_def = tf.GraphDef()	
    graph_def.ParseFromString(f.read())	
    _ = tf.import_graph_def(graph_def, name='')	
	
	
with tf.Session() as sess:

    video_capture = cv2.VideoCapture(video_path) 
    
    i = 0
    while True:  
        frame = video_capture.read()[1] 
        frameId = video_capture.get(1) 
        
        if (0 == 0):  
            i = i + 1
            cv2.imwrite(filename="screens/"+str(i)+"alpha.png", img=frame); 
            image_data = tf.gfile.FastGFile("screens/"+str(i)+"alpha.png", 'rb').read() 
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor, \
                     {'DecodeJpeg/contents:0': image_data})     
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                print('%s (score = %.5f)' % (human_string, score))
            print ("\n\n")
            cv2.imshow("image", frame)  
            cv2.waitKey(1)  

    video_capture.release() 
    cv2.destroyAllWindows() 