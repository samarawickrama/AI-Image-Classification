import tensorflow as tf 
import sys
import shutil
import os.path
import glob

image_list = []
for image in glob.glob('input/*.jpg'): #assuming jpg

    tf.reset_default_graph() # Reset Graph
    image_path = image
    
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
        
        index = 1
        classFlag = 0
        for node_id in top_k:
            index = index + 1
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
            
            if (score > 0.85):
                classFlag = 1
                output_path = os.path.join('output', human_string)
                if not os.path.isdir(output_path):
                    os.makedirs (output_path)
                shutil.copy2(image_path, output_path)
            else:
                if index == len(top_k) and classFlag == 0:
                    output_path = os.path.join('output', 'unknown')
                    if not os.path.isdir(output_path):
                        os.makedirs (output_path)
                    shutil.copy2(image_path, output_path)
                
    
