import tensorflow as tf
import keras.backend as K

def centernet_loss(classes=10,
                   alpha_pos=2.0,
                   alpha_neg=2.0,
                   beta=4.0,
                   neg_weight=0.5,
                   scales_weight=10.,
                   img_h=640,
                   img_w=1024):
    def loss(y_true, y_pred):

        hm = y_pred[:,:,:,:classes]
        hm_t = y_true[:,:,:,:classes]

        sc = y_pred[:,:,:,classes:]
        sc_t = y_true[:,:,:,classes:]

        Lk_pos_mask = tf.where(tf.equal(hm_t, 1.0), tf.ones_like(hm_t), tf.zeros_like(hm_t))
        Lk_neg_mask = tf.where(tf.less(hm_t, 1.0), tf.ones_like(hm_t), tf.zeros_like(hm_t))

        hm = tf.clip_by_value(hm, K.epsilon(), 1. - K.epsilon())

        N = tf.reduce_sum(Lk_pos_mask)

        Lk_pos = tf.zeros_like(hm) - tf.pow((tf.ones_like(hm) - hm), alpha_pos) * tf.log(hm) * Lk_pos_mask
        Lk_neg = tf.zeros_like(hm) - tf.pow((tf.ones_like(hm_t) - hm_t), beta) * tf.pow(hm, alpha_neg) * tf.log(tf.ones_like(hm) - hm) * Lk_neg_mask

        Lk_pos_red = tf.reduce_sum(Lk_pos) / N
        Lk_neg_red = neg_weight * tf.reduce_sum(Lk_neg) / N

        Lk = Lk_pos_red + Lk_neg_red

        Lsc_pos_mask = tf.reduce_sum(Lk_pos_mask, axis=3, keepdims=True)

        Lsc = tf.multiply(tf.abs(tf.subtract(sc_t, sc)), Lsc_pos_mask)
        Lsc_red = tf.reduce_sum(Lsc) / N

        L = Lk + scales_weight * Lsc_red

        return L
    return loss

def center_pos_loss(classes=10,
                   alpha=2.0,
                   beta=4.0):
    def pos_loss(y_true, y_pred):

        hm = y_pred[:,:,:,:classes]
        hm_t = y_true[:,:,:,:classes]

        Lk_pos_mask = tf.where(tf.equal(hm_t, 1.0), tf.ones_like(hm_t), tf.zeros_like(hm_t))

        hm = tf.clip_by_value(hm, K.epsilon(), 1.0 - K.epsilon())

        N = tf.reduce_sum(Lk_pos_mask)

        Lk_pos = tf.zeros_like(hm) - tf.pow((tf.ones_like(hm) - hm), alpha) * tf.log(hm) * Lk_pos_mask

        Lk = tf.reduce_sum(Lk_pos) / N

        L = Lk

        return L
    return pos_loss

def center_neg_loss(classes=10,
                   alpha=2.0,
                   beta=4.0):
    def neg_loss(y_true, y_pred):

        hm = y_pred[:,:,:,:classes]
        hm_t = y_true[:,:,:,:classes]

        Lk_pos_mask = tf.where(tf.equal(hm_t, 1.0), tf.ones_like(hm_t), tf.zeros_like(hm_t))
        Lk_neg_mask = tf.where(tf.less(hm_t, 1.0), tf.ones_like(hm_t), tf.zeros_like(hm_t))

        hm = tf.clip_by_value(hm, K.epsilon(), 1.0 - K.epsilon())

        N = tf.reduce_sum(Lk_pos_mask)

        Lk_neg = tf.zeros_like(hm) - tf.pow((tf.ones_like(hm_t) - hm_t), beta) * tf.pow(hm, alpha) * tf.log(tf.ones_like(hm) - hm) * Lk_neg_mask

        Lk = tf.reduce_sum(Lk_neg) / N

        L = Lk

        return L
    return neg_loss

def reg_loss(classes=10,
                   alpha_pos=2.0,
                   alpha_neg=2.0,
                   beta=4.0,
                   neg_weight=1.0,
                   scales_weight=10.0):
    def regr_loss(y_true, y_pred):

        hm_t = y_true[:,:,:,:classes]

        sc = y_pred[:,:,:,classes:]
        sc_t = y_true[:,:,:,classes:]

        Lk_pos_mask = tf.where(tf.equal(hm_t, 1.0), tf.ones_like(hm_t), tf.zeros_like(hm_t))
        Lk_neg_mask = tf.where(tf.less(hm_t, 1.0), tf.ones_like(hm_t), tf.zeros_like(hm_t))

        N = tf.reduce_sum(Lk_pos_mask)

        Lsc_pos_mask = tf.reduce_sum(Lk_pos_mask, axis=3, keepdims=True)

        Lsc = tf.multiply(tf.abs(tf.subtract(sc_t, sc)), Lsc_pos_mask)
        Lsc_red = tf.reduce_sum(Lsc) / N

        L = scales_weight * Lsc_red

        return L
    return regr_loss