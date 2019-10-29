import tensorflow as tf

#生成带序号的网格
def _create_mesh_xy(batch_size,grid_h,grid_w,n_box):
    mesh_x = tf.cast(tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1, 1)), tf.float32)
    mesh_y = tf.transpose(mesh_x, (0, 2, 1, 3, 4))
    mesh_xy = tf.tile(tf.concat([mesh_x, mesh_y], -1), [batch_size, 1, 1, n_box, 1])
    return mesh_xy

#将网格信息融入坐标，置信度做sigmoid。并重新组合
def adjust_pred_tensor(y_pred):
    grid_offset=_create_mesh_xy(*y_pred.shape[:4])
    # 计算该尺度矩阵上的坐标sigma(t_xy) + c_xy
    pred_xy=grid_offset+tf.sigmoid(y_pred[...,:2])
    # 取出预测物体的尺寸t_wh
    pred_wh=y_pred[...,2:4]
    # 对分类概率（置信度）做sigmoid转化
    pred_conf=tf.sigmoid(y_pred[...,4])
    # 取出分类结果
    pred_classes=y_pred[...,5:]

    #重新组合
    preds=tf.concat([pred_xy,pred_wh,tf.expand_dims(pred_conf,axis=-1),pred_classes],axis=-1)

    return preds


#生成一个矩阵。每个格子里放有3个候选框
def _create_mesh_anchor(anchors, batch_size, grid_h, grid_w, n_box):
    mesh_anchor = tf.tile(anchors, [batch_size * grid_h * grid_w])
    mesh_anchor = tf.reshape(mesh_anchor, [batch_size, grid_h, grid_w, n_box, 2])  # 每个候选框有2个值
    mesh_anchor = tf.cast(mesh_anchor, tf.float32)
    return mesh_anchor

def conf_delta_tensor(y_true, y_pred, anchors, ignore_thresh):
    pred_box_xy,pred_box_wh,pred_box_conf=y_pred[...,:2],y_pred[...,2:4],y_pred[...,4]
    #带有候选框的格子矩阵
    anchors_grid=_create_mesh_anchor(anchors,*y_pred.shape[:4])
    true_wh=y_true[:,:,:,:,2:4]
    true_wh=anchors_grid*tf.exp(true_wh)
    # 还原真实尺寸，高和宽
    true_wh=true_wh*tf.expand_dims(y_true[:,:,:,:,4],4)
    # y_pred.shape[3]为候选框个数
    anchors_=tf.constant(anchors,dtype="float",shape=[1,1,1,y_pred.shape[3],2])
    #获取中心点
    true_xy=y_true[...,0:2]
    true_wh_half=true_wh/2.
    true_mins=true_xy-true_wh_half#左上角
    true_maxes=true_xy+true_wh_half#右下角

    pred_xy=pred_box_xy
    pred_wh=tf.exp(pred_box_wh)*anchors_

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half  # 计算起始坐标
    pred_maxes = pred_xy + pred_wh_half  # 计算尾部坐标

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)

    #计算重叠面积
    intersect_wh=tf.maximum(intersect_maxes-intersect_mins,0.)
    intersect_areas=intersect_wh[...,0]*intersect_wh[...,1]

    true_area=true_wh[...,0]*true_wh[...,1]
    pred_area=pred_wh[...,0]*pred_wh[...,1]

    #计算不重叠面积
    union_area=pred_area+true_area-intersect_areas
    best_ious=tf.truediv(intersect_areas,union_area)#计算iou
    #iou小于阈值作为负向的loss
    conf_delta=pred_box_conf*tf.cast(best_ious<ignore_thresh,tf.float32)

    return conf_delta



def wh_scale_tensor(true_box_wh, anchors, image_size):
    image_size_=tf.reshape(tf.cast(image_size,tf.float32),[1,1,1,1,2])
    anchors_=tf.constant(anchors,dtype='float',shape=[1,1,1,3,2])

    #计算高和宽缩放范围
    wh_scale=tf.exp(true_box_wh)*anchors_/image_size_
    #物体尺寸占整个图片的面积比
    wh_scale=tf.expand_dims(2-wh_scale[...,0]*wh_scale[...,1],axis=4)

    return wh_scale

#位置loss为box之差乘缩放比，所得的结果，再进行平方求和
def loss_coord_tensor(object_mask, pred_box, true_box, wh_scale, xywh_scale):
    xy_delta=object_mask*(pred_box-true_box)*wh_scale*xywh_scale
    loss_xy=tf.reduce_sum(tf.square(xy_delta),list(range(1,5)))

    return loss_xy

def loss_conf_tensor(object_mask, pred_box_conf, true_box_conf, obj_scale, noobj_scale, conf_delta):
    object_mask_ = tf.squeeze(object_mask, axis=-1)
    conf_delta=object_mask_*(pred_box_conf-true_box_conf) * obj_scale + (1-object_mask_) * conf_delta * noobj_scale
    loss_conf=tf.reduce_sum(tf.square(conf_delta),list(range(1,4)))
    return loss_conf

def loss_class_tensor(object_mask, pred_box_class, true_box_class, class_scale):
    true_box_class_ = tf.cast(true_box_class, tf.int64)

    class_delta = object_mask *\
                  tf.expand_dims(tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_box_class_, logits=pred_box_class), 4) * \
                  class_scale

    loss_class = tf.reduce_sum(class_delta, list(range(1, 5)))
    return loss_class

ignore_thresh=0.5
grid_scale=1
obj_scale=5
noobj_scale=1
xywh_scale=1
class_scale=1
def lossCalculator(y_true, y_pred, anchors,image_size):
    y_pred=tf.reshape(y_pred,y_true.shape)

    object_mask=tf.expand_dims(y_true[...,4],4)
    preds=adjust_pred_tensor(y_pred)

    conf_delta = conf_delta_tensor(y_true, preds, anchors, ignore_thresh)
    wh_scale = wh_scale_tensor(y_true[..., 2:4], anchors, image_size)

    loss_box = loss_coord_tensor(object_mask, preds[..., :4], y_true[..., :4], wh_scale, xywh_scale)
    loss_conf = loss_conf_tensor(object_mask, preds[..., 4], y_true[..., 4], obj_scale, noobj_scale, conf_delta)
    loss_class = loss_class_tensor(object_mask, preds[..., 5:], y_true[..., 5:], class_scale)
    loss = loss_box + loss_conf + loss_class

    return loss * grid_scale


def loss_fn(list_y_trues, list_y_preds,anchors,image_size):
    inputanchors = [anchors[12:], anchors[6:12], anchors[:6]]

    losses = [lossCalculator(list_y_trues[i], list_y_preds[i], inputanchors[i], image_size) for i in
              range(len(list_y_trues))]

    return tf.sqrt(tf.reduce_sum(losses))  # 将三个矩阵的loss相加再开平方


if __name__=="__main__":
    xy=_create_mesh_xy(1,8,8,3)
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    m_xy=sess.run(xy)
    print(m_xy)
