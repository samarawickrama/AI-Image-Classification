#!C:/Users/SAMARAM/AppData/Local/Continuum/Anaconda3/python.exe

print("content-type: text/html\n\n" )

import os
import cgi, cgitb
import shutil

cgitb.enable()

#print("Uploading<br>")

form = cgi.FieldStorage()
filedata = form['upload']

with open("input\\" + filedata.filename + ".jpg", "wb") as fout:
    shutil.copyfileobj(filedata.file, fout, 100000)
	
#print("Upload Completed<br>")

######################################################################

print("Classification Results<br>")
print("================<br>")

import tensorflow as tf 
import sys

image_path = 'input/' + filedata.filename + '.jpg'
print("<img src='" + image_path + "' height=\"100\"><br>")

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrain_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrain_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f) <br>' % (human_string, score))

print("<font color=\"red\"><br> Note: <br>The algorithm is currently trained on specific image categories (i.e., babies, city, dessert, flowers, food, hills, music, nature, night sky, portraits, sport, underwater, vehicles). It output the image classification and the level of confidence. Any image outside those categories will not correctly categorize (it will process the image but the confidence will be low).</font>")

