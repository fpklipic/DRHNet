import tensorflow as tf
import numpy as np
import Net
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='dehaze_outdoor', help='dehaze_outdoor| dehaze_indoor | derain')
parser.add_argument('--indir', default='samples/1.jpg',help='the path of the input image')
parser.add_argument('--showShortcut', type=bool,default=False,help='True | False')
opt = parser.parse_args()

def Test(path):
    test_img = cv2.imread(path)
    (h1,w1,d) = test_img.shape
    disH = 0
    disW = 0
    sizeThreshold = 0
    if opt.task == 'dehaze_outdoor':
        sizeThreshold = 550
    elif opt.task == 'dehaze_indoor':
        sizeThreshold = 620
    elif opt.task == 'derain':
        sizeThreshold = 512
    else:
        print('The task is undefined!')
        return
    if h1 <= sizeThreshold and w1<= sizeThreshold:
        disH = int((sizeThreshold-h1)/2)
        disW = int((sizeThreshold-w1)/2)
        test_img = cv2.copyMakeBorder(test_img, disH,sizeThreshold-h1-disH,disW,sizeThreshold-w1-disW, cv2.BORDER_REFLECT)
    (h,w,d) = test_img.shape
    test_size_h = h
    test_size_w = w
    global_step = tf.Variable(0)
    input_X = tf.placeholder(tf.float32, [1, test_size_h,test_size_w,3])
    net = Net.Net()
    source,shortcut,pre = net.model(input_X,training=True)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=3) 
    print ("GRAPH READY")
    sess = tf.Session()
    sess.run(init)
    if opt.task == 'dehaze_outdoor':
        saver.restore(sess, "models/model_dehaze_outdoor")
    elif opt.task == 'dehaze_indoor':
        saver.restore(sess, "models/model_dehaze_indoor")
    elif opt.task == 'derain':
        saver.restore(sess, "models/model_dehaze_DID")
    X_num = int(w / test_size_w)
    Y_num = int(h /test_size_h)
    
    for i in range(Y_num):
        for j in range(X_num):
            block = test_img[i*test_size_h:(i+1)*test_size_h,j*test_size_w:(j+1)*test_size_w,:]
            print(block.shape)
            if block.shape != (test_size_h,test_size_w,3):
                continue
            block = np.reshape(block,[-1,test_size_h,test_size_w,3])
            input_X_img,shortcut_img,pre_img = sess.run([source,shortcut,pre],feed_dict={input_X: block})
            pre_img = np.reshape(pre_img,[test_size_h,test_size_w,3])
            shortcut_img = np.reshape(shortcut_img,[test_size_h,test_size_w,3])
            if opt.showShortcut == True:
                for i1 in range(shortcut_img.shape[0]):
                    for j1 in range(shortcut_img.shape[1]):
                        for k1 in range(shortcut_img.shape[2]):
                            shortcut_img[i1][j1][k1] = min(0,shortcut_img[i1][j1][k1])
                            shortcut_img[i1][j1][k1] = -shortcut_img[i1][j1][k1]
            test_img[i*test_size_h:(i+1)*test_size_h,j*test_size_w:(j+1)*test_size_w,:]= pre_img
    if opt.showShortcut == True:
        shortcut_img=shortcut_img.astype(np.uint8)
        cv2.imwrite("./output/shortcut_"+(path.split('/')[-1]),shortcut_img[disH:disH+h1,disW:disW+w1,:])
    cv2.imwrite("./output/"+opt.task+'_'+(path.split('/')[-1]),test_img[disH:disH+h1,disW:disW+w1,:])
if __name__ == '__main__':
    Test(opt.indir)
