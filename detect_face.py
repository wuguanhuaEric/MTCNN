import numpy as np  
import tensorflow as tf  
from model import fcn_12_detect  
  
  
def py_nms(dets, thresh, mode="Union"):  
    """ 
    greedily select boxes with high confidence 
    keep boxes overlap <= thresh 
    rule out overlap > thresh 
    :param dets: [[x1, y1, x2, y2 score]] 
    :param thresh: retain overlap <= thresh 
    :return: indexes to keep 
    """  
    if len(dets) == 0:  
        return []  
    x1 = dets[:, 0]  
    y1 = dets[:, 1]  
    x2 = dets[:, 2]  
    y2 = dets[:, 3]  
    scores = dets[:, 4]  
  
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
    order = scores.argsort()[::-1]  
  
    keep = []  
    while order.size > 0:  
        i = order[0]  
        keep.append(i)  
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        if mode == "Union":  
            ovr = inter / (areas[i] + areas[order[1:]] - inter)  
        elif mode == "Minimum":  
            ovr = inter / np.minimum(areas[i], areas[order[1:]])  
  
        inds = np.where(ovr <= thresh)[0]  
        order = order[inds + 1]  
  
    return dets[keep]  
  
def image_preprocess(img):  
  
    img = (img - 127.5)*0.0078125  
    '''''m = img.mean() 
    s = img.std() 
    min_s = 1.0/(np.sqrt(img.shape[0]*img.shape[1]*img.shape[2])) 
    std = max(min_s, s)   
    img = (img-m)/std'''  
  
    return img  
  
  
  
def slide_window(img, window_size, stride):  
    # 对构建的金字塔图片，滑动窗口。  
    # img：图片， window_size：滑动窗的大小，stride：步长。  
      
    window_list = []  
      
    w = img.shape[1]  
    h = img.shape[0]  
  
    if w<=window_size+stride or h<=window_size+stride:  
        return None  
    if len(img.shape)!=3:  
        return None  
      
    for i in range(int((w-window_size)/stride)):  
        for j in range(int((h-window_size)/stride)):  
            box = [j*stride, i*stride, j*stride+window_size, i*stride+window_size]  
              
            window_list.append(box)  
  
return img, np.asarray(window_list)  
  
  
def pyramid(image, f, window_size):  
    # 构建图像的金字塔，以便进行多尺度滑动窗口  
    # image：输入图像，f：缩放的尺度， window_size：滑动窗大小。  
    w = image.shape[1]  
    h = image.shape[0]  
    img_ls = []  
    while( w > window_size and h > window_size):  
        img_ls.append(image)  
        w = int(w * f)  
        h = int(h * f)  
        image = cv2.resize(image, (w, h))  
    return img_ls  
  
def min_face(img, F, window_size, stride):  
    # img：输入图像，F：最小人脸大小， window_size：滑动窗，stride：滑动窗的步长。  
    h, w, _ = img.shape  
    w_re = int(float(w)*window_size/F)  
    h_re = int(float(h)*window_size/F)  
    if w_re<=window_size+stride or h_re<=window_size+stride:  
        print (None)  
    # 调整图片大小的时候注意参数，千万不要写反了  
    # 根据最小人脸缩放图片  
    img = cv2.resize(img, (w_re, h_re))  
    return img  
  
  
  
if __name__ = "__main__":  
      
    image = cv2.imread('images/1.jpg')  
    h,w,_ = image.shape  
      
    ......  
    # 调参的参数  
    IMAGE_SIZE = 12  
    # 步长  
    stride = 2  
    # 最小人脸大小  
    F = 40  
    # 构建金字塔的比例  
    ff = 0.8  
    # 概率多大时判定为人脸？  
    p_12 = 0.8  
    p_24 = 0.8  
    # nms  
    overlapThresh_12 = 0.7  
    overlapThresh_24 = 0.3  
    ......  
    # 加载 model  
    net_12 = fcn_12_detect()  
    net_12_vars = [v for v in tf.trainable_variables() if v.name.startswith('net_12')]  
    saver_net_12 = tf.train.Saver(net_12_vars)  
    sess = tf.Session()  
    sess.run(tf.initialize_all_variables())  
    saver_net_12.restore(sess, 'model/12-net/model_net_12-123200')  
    # net_24...  
    ......  
    # 需要检测的最小人脸  
    image_ = min_face(image, F, IMAGE_SIZE, stride)  
    ......  
    # 金字塔  
    pyd = pyramid(np.array(image_), ff, IMAGE_SIZE)  
    ......  
    # net-12  
    window_after_12 = []  
    for i, img in enumerate(pyd):  
        # 滑动窗口  
        slide_return = slide_window(img, IMAGE_SIZE, stride)  
        if slide_return is None:  
            break  
        img_12 = slide_return[0]  
        window_net_12 = slide_return[1]  
        w_12 = img_12.shape[1]  
        h_12 = img_12.shape[0]  
          
        patch_net_12 = []  
        for box in window_net_12:  
            patch = img_12[box[0]:box[2], box[1]:box[3], :]  
            # 做归一化处理  
            patch = image_preprocess(patch)  
            patch_net_12.append(patch)  
        patch_net_12 = np.array(patch_net_12)  
      
        # 预测人脸  
        pred_cal_12 = sess.run(net_12['pred'], feed_dict={net_12['imgs']: patch_net_12})  
        window_net = window_net_12  
        # print (pred_cal_12)  
        windows = []  
        for i, pred in enumerate(pred_cal_12):  
            # 概率大于0.8的判定为人脸。  
            s = np.where(pred[1]>p_12)[0]  
            if len(s)==0:  
                continue  
            #保存窗口位置和概率。  
            windows.append([window_net[i][0],window_net[i][1],window_net[i][2],window_net[i][3],pred[1]])  
          
        # 按照概率值 由大到小排序  
        windows = np.asarray(windows)  
        windows = py_nms(windows, overlapThresh_12, 'Union')  
        window_net = windows  
        for box in window_net:  
            lt_x = int(float(box[0])*w/w_12)  
            lt_y = int(float(box[1])*h/h_12)  
            rb_x = int(float(box[2])*w/w_12)  
            rb_y = int(float(box[3])*h/h_12)  
            p_box = box[4]  
            window_after_12.append([lt_x, lt_y, rb_x, rb_y, p_box])  
    # 按照概率值 由大到小排序  
    # window_after_12 = np.asarray(window_after_12)  
    # window_net = py_nms(window_after_12, overlapThresh_12, 'Union')  
    window_net = window_after_12  
    print (window_net)  
      
    # net-24  
    windows_24 = []  
    if window_net == []:  
        print "windows is None!"  
    if window_net != []:  
        patch_net_24 = []  
        img_24 = image  
        for box in window_net:  
            patch = img_24[box[0]:box[2], box[1]:box[3], :]  
            patch = cv2.resize(patch, (24, 24))  
            # 做归一化处理  
            patch = image_preprocess(patch)  
            patch_net_24.append(patch)  
        # 预测人脸  
        pred_net_24 = sess.run(net_24['pred'], feed_dict={net_24['imgs']: patch_net_24})  
        print (pred_net_24)  
        window_net = window_net  
        # print (pred_net_24)  
        for i, pred in enumerate(pred_net_24):  
            s = np.where(pred[1]>p_24)[0]  
            if len(s)==0:  
                continue  
            windows_24.append([window_net[i][0],window_net[i][1],window_net[i][2],window_net[i][3],pred[1]])  
        # 按照概率值 由大到小排序  
        windows_24 = np.asarray(windows_24)  
        #window_net = nms_max(windows_24, overlapThresh=0.7)  
        window_net = py_nms(windows_24, overlapThresh_24, 'Union')  
  
  
    if window_net == []:  
        print "windows is None!"  
    if window_net != []:  
        print(window_net.shape)  
        for box in window_net:  
            #ImageDraw.Draw(image).rectangle((box[1], box[0], box[3], box[2]), outline = "red")  
            cv2.rectangle(image, (int(box[1]),int(box[0])), (int(box[3]),int(box[2])), (0, 255, 0), 2)  
    cv2.imwrite("images/face_img.jpg", image)  
    cv2.imshow("face detection", image)  
    cv2.waitKey(10000)  
    cv2.destroyAllWindows()  
      
  
    coord.request_stop()  
    coord.join(threads)  
  
    sess.close()  