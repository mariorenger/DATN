import tensorflow.compat.v1.keras as keras
from tensorflow.compat.v1.linalg import einsum
from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras import backend as K
import tensorflow as tf
import numpy
import sys

class AttLayer(layers.Layer):
    """Soft alignment attention implement.

    Attributes:
        dim (int): attention hidden dim
    """

    def __init__(self, dim=200, seed=0, **kwargs):
        """Initialization steps for AttLayer2.

        Args:
            dim (int): attention hidden dim
        """

        self.dim = dim
        self.seed = seed
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """Initialization for variables in AttLayer2
        There are there variables in AttLayer2, i.e. W, b and q.

        Args:
            input_shape (object): shape of input tensor.
        """

        assert len(input_shape) == 3
        dim = self.dim
        self.W = self.add_weight(
            name="W",
            shape=(int(input_shape[-1]), dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(dim,),
            initializer=keras.initializers.Zeros(),
            trainable=True,
        )
        self.q = self.add_weight(
            name="q",
            shape=(dim, 1),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )

        self.ln = tf.keras.layers.Dense(int(input_shape[-1]), activation='relu',trainable=False)
        self.ln_out = tf.keras.layers.Dense(1, activation='sigmoid', trainable=False)

        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def pooling_along_time(self, rep_tensor, rep_mask, pooling_method='mean',
                   keep_dim=False, keep_len=False, name=None):
        bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
        ivec = rep_tensor.get_shape().as_list()[2]
        with tf.name_scope(name or '%s_pooling_along_time' % pooling_method):
            rep_mask_tiled = tf.tile(tf.expand_dims(rep_mask, 2), [1, 1, ivec])
            denominator = tf.reduce_sum(tf.cast(rep_mask, tf.int32), 1, keepdims=True)  # bs, 1
            denominator_tiled = tf.tile(denominator, [1, ivec])
            if pooling_method == 'mean':
                rep_tensor_zero = tf.where(rep_mask_tiled, rep_tensor, tf.zeros_like(rep_tensor))  # bs,sl,vec
                rep_sum = tf.reduce_sum(rep_tensor_zero, 1)  # bs, vec
                denominator = tf.where(tf.equal(denominator, 0), tf.ones_like(denominator), denominator)
                res = rep_sum / tf.cast(denominator, tf.float32)
            else:
                raise RuntimeError('no pooling method %s' % pooling_method)

            if keep_dim:
                res = tf.expand_dims(res, 1)
                if keep_len:
                    res = tf.tile(res, [1, sl, 1])
            return res

    def sequence_conditional_feature(self, rep_tensor, rep_mask):
        rep_tensor_pooling = self.pooling_along_time(rep_tensor, rep_mask, 'mean', True, True)
        rep_tensor_new = tf.concat([rep_tensor, rep_tensor_pooling, rep_tensor * rep_tensor_pooling], -1)
        return rep_tensor_new

    def call(self, inputs, mask=None, **kwargs):
        """Core implemention of soft attention

        Args:
            inputs (object): input tensor.

        Returns:
            object: weighted sum of input tensors.
        """

        attention = K.tanh(K.dot(inputs, self.W) + self.b)
        attention = K.dot(attention, self.q)

        attention = K.squeeze(attention, axis=2)

        rep_mask = tf.ones_like(inputs[:, :, -1], dtype=tf.bool)

        rss = self.sequence_conditional_feature(inputs, rep_mask)

        z = self.ln(rss)
        z = self.ln_out(z)

        rewards_raw = - tf.math.log(z)
        cost_rl = - tf.reduce_mean(rewards_raw * tf.reduce_sum(z, 1), name='cost_rl')

        z = K.squeeze(z, axis=2) > 0.5

        self.rl_loss = cost_rl

        M = tf.cast(z, tf.float32)

        attention = tf.multiply(attention, M)

        if mask is None:
            attention = K.exp(attention)
        else:
            attention = K.exp(attention) * K.cast(mask, dtype="float32")

        attention_weight = attention / (
            K.sum(attention, axis=-1, keepdims=True) + K.epsilon()
        )

        attention_weight = K.expand_dims(attention_weight)
        weighted_input = inputs * attention_weight
        return K.sum(weighted_input, axis=1)

    def compute_mask(self, input, input_mask=None):
        """Compte output mask value

        Args:
            input (object): input tensor.
            input_mask: input mask

        Returns:
            object: output mask.
        """
        return None

    def compute_output_shape(self, input_shape):
        """Compute shape of output tensor

        Args:
            input_shape (tuple): shape of input tensor.

        Returns:
            tuple: shape of output tensor.
        """
        return input_shape[0], input_shape[-1]


class AttLayer2(layers.Layer):
    """Soft alignment attention implement.

    Attributes:
        dim (int): attention hidden dim
    """

    def __init__(self, dim=200, seed=0, **kwargs):
        """Initialization steps for AttLayer2.

        Args:
            dim (int): attention hidden dim
        """

        self.dim = dim
        self.seed = seed
        super(AttLayer2, self).__init__(**kwargs)
        self.att_w = []

    def build(self, input_shape):
        """Initialization for variables in AttLayer2
        There are there variables in AttLayer2, i.e. W, b and q.

        Args:
            input_shape (object): shape of input tensor.
        """

        assert len(input_shape) == 3
        dim = self.dim
        self.W = self.add_weight(
            name="W",
            shape=(int(input_shape[-1]), dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(dim,),
            initializer=keras.initializers.Zeros(),
            trainable=True,
        )
        self.q = self.add_weight(
            name="q",
            shape=(dim, 1),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        super(AttLayer2, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, inputs, mask=None, **kwargs):
        """Core implemention of soft attention

        Args:
            inputs (object): input tensor.

        Returns:
            object: weighted sum of input tensors.
        """

        attention = K.tanh(K.dot(inputs, self.W) + self.b)
        attention = K.dot(attention, self.q)

        attention = K.squeeze(attention, axis=2)

        if mask is None:
            attention = K.exp(attention)
        else:
            attention = K.exp(attention) * K.cast(mask, dtype="float32")

        attention_weight = attention / (
            K.sum(attention, axis=-1, keepdims=True) + K.epsilon()
        )
        self.att_w = attention_weight 

        attention_weight = K.expand_dims(attention_weight)
        weighted_input = inputs * attention_weight
        return K.sum(weighted_input, axis=1)

    def compute_mask(self, input, input_mask=None):
        """Compte output mask value

        Args:
            input (object): input tensor.
            input_mask: input mask

        Returns:
            object: output mask.
        """
        return None

    def compute_output_shape(self, input_shape):
        """Compute shape of output tensor

        Args:
            input_shape (tuple): shape of input tensor.

        Returns:
            tuple: shape of output tensor.
        """
        return input_shape[0], input_shape[-1]


class SelfAttention2(layers.Layer):
    """Multi-head self attention implement.

    Args:
        multiheads (int): The number of heads.
        head_dim (object): Dimention of each head.
        mask_right (boolean): whether to mask right words.

    Returns:
        object: Weighted sum after attention.
    """

    def __init__(self, multiheads, head_dim, seed=0, mask_right=False, **kwargs):
        """Initialization steps for AttLayer2.

        Args:
            multiheads (int): The number of heads.
            head_dim (object): Dimension of each head.
            mask_right (boolean): Whether to mask right words.
        """

        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        self.seed = seed
        super(SelfAttention2, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """Compute shape of output tensor.

        Returns:
            tuple: output shape tuple.
        """

        return (input_shape[0][0], input_shape[0][1], self.output_dim)

    def build(self, input_shape):
        """Initialization for variables in SelfAttention.
        There are three variables in SelfAttention, i.e. WQ, WK ans WV.
        WQ is used for linear transformation of query.
        WK is used for linear transformation of key.
        WV is used for linear transformation of value.

        Args:
            input_shape (object): shape of input tensor.
        """

        self.WQ = self.add_weight(
            name="WQ",
            shape=(int(input_shape[0][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        self.WK = self.add_weight(
            name="WK",
            shape=(int(input_shape[1][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        self.WV = self.add_weight(
            name="WV",
            shape=(int(input_shape[2][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        super(SelfAttention2, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode="add"):
        """Mask operation used in multi-head self attention

        Args:
            seq_len (object): sequence length of inputs.
            mode (str): mode of mask.

        Returns:
            object: tensors after masking.
        """

        if seq_len is None:
            return inputs
        else:
            mask = K.one_hot(indices=seq_len[:, 0], num_classes=K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, axis=1)

            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)

            if mode == "mul":
                return inputs * mask
            elif mode == "add":
                return inputs - (1 - mask) * 1e12

    def call(self, QKVs):
        """Core logic of multi-head self attention.

        Args:
            QKVs (list): inputs of multi-head self attention i.e. query, key and value.

        Returns:
            object: ouput tensors.
        """
        if len(QKVs) == 3:
            Q_seq, K_seq, V_seq = QKVs
            Q_len, V_len = None, None
        elif len(QKVs) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = QKVs
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(
            Q_seq, shape=(-1, K.shape(Q_seq)[1], self.multiheads, self.head_dim)
        )
        Q_seq = K.permute_dimensions(Q_seq, pattern=(0, 2, 1, 3))

        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(
            K_seq, shape=(-1, K.shape(K_seq)[1], self.multiheads, self.head_dim)
        )
        K_seq = K.permute_dimensions(K_seq, pattern=(0, 2, 1, 3))

        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(
            V_seq, shape=(-1, K.shape(V_seq)[1], self.multiheads, self.head_dim)
        )
        V_seq = K.permute_dimensions(V_seq, pattern=(0, 2, 1, 3))

        A = einsum("abij, abkj -> abik", Q_seq, K_seq) / K.sqrt(
            K.cast(self.head_dim, dtype="float32")
        )
        A = K.permute_dimensions(
            A, pattern=(0, 3, 2, 1)
        )  # A.shape=[batch_size,K_sequence_length,Q_sequence_length,self.multiheads]

        A = self.Mask(A, V_len, "add")
        A = K.permute_dimensions(A, pattern=(0, 3, 2, 1))

        if self.mask_right:
            ones = K.ones_like(A[:1, :1])
            lower_triangular = K.tf.matrix_band_part(ones, num_lower=-1, num_upper=0)
            mask = (ones - lower_triangular) * 1e12
            A = A - mask
        A = K.softmax(A)

        O_seq = einsum("abij, abjk -> abik", A, V_seq)
        O_seq = K.permute_dimensions(O_seq, pattern=(0, 2, 1, 3))

        O_seq = K.reshape(O_seq, shape=(-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, "mul")
        return O_seq

    def get_config(self):
        """add multiheads, multiheads and mask_right into layer config.

        Returns:
            dict: config of SelfAttention layer.
        """
        config = super(SelfAttention2, self).get_config()
        config.update(
            {
                "multiheads": self.multiheads,
                "head_dim": self.head_dim,
                "mask_right": self.mask_right,
            }
        )
        return config


def PersonalizedAttentivePooling(dim1, dim2, dim3, seed=0):
    """Soft alignment attention implement.

    Attributes:
        dim1 (int): first dimention of value shape.
        dim2 (int): second dimention of value shape.
        dim3 (int): shape of query

    Returns:
        object: weighted summary of inputs value.
    """
    vecs_input = keras.Input(shape=(dim1, dim2), dtype="float32")
    query_input = keras.Input(shape=(dim3,), dtype="float32")

    user_vecs = layers.Dropout(0.2)(vecs_input)
    user_att = layers.Dense(
        dim3,
        activation="tanh",
        kernel_initializer=keras.initializers.glorot_uniform(seed=seed),
        bias_initializer=keras.initializers.Zeros(),
    )(user_vecs)
    user_att2 = layers.Dot(axes=-1)([query_input, user_att])
    user_att2 = layers.Activation("softmax")(user_att2)
    user_vec = layers.Dot((1, 1))([user_vecs, user_att2])

    model = keras.Model([vecs_input, query_input], user_vec)
    return model

class SelfAttention(layers.Layer):
    """Multi-head self attention implement.

    Args:
        multiheads (int): The number of heads.
        head_dim (object): Dimention of each head.
        mask_right (boolean): whether to mask right words.

    Returns:
        object: Weighted sum after attention.
    """

    def __init__(self, multiheads, head_dim, seed=0, mask_right=False, **kwargs):
        """Initialization steps for AttLayer2.

        Args:
            multiheads (int): The number of heads.
            head_dim (object): Dimension of each head.
            mask_right (boolean): Whether to mask right words.
        """

        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        self.seed = seed
        super(SelfAttention, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """Compute shape of output tensor.

        Returns:
            tuple: output shape tuple.
        """

        return (input_shape[0][0], input_shape[0][1], self.output_dim)

    def build(self, input_shape):
        """Initialization for variables in SelfAttention.
        There are three variables in SelfAttention, i.e. WQ, WK ans WV.
        WQ is used for linear transformation of query.
        WK is used for linear transformation of key.
        WV is used for linear transformation of value.

        Args:
            input_shape (object): shape of input tensor.
        """

        self.WQ = self.add_weight(
            name="WQ",
            shape=(int(input_shape[0][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        self.WK = self.add_weight(
            name="WK",
            shape=(int(input_shape[1][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        self.WV = self.add_weight(
            name="WV",
            shape=(int(input_shape[2][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )

        self.WM = self.add_weight(
            name="WM",
            shape=(int(input_shape[-1][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=False,
        )

        self.ln_1k = tf.keras.layers.Dense(int(input_shape[-1][-1]), trainable=False)
        self.ln_2k = tf.keras.layers.Dense(1, activation="sigmoid",  trainable=False)

        self.ln_1q = tf.keras.layers.Dense(int(input_shape[-1][-1]), trainable=False)
        self.ln_2q = tf.keras.layers.Dense(1, activation="sigmoid",  trainable=False)

        super(SelfAttention, self).build(input_shape)

    def pooling_along_time(self, rep_tensor, rep_mask, pooling_method='mean',
                   keep_dim=False, keep_len=False, name=None):
        bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
        ivec = rep_tensor.get_shape().as_list()[2]
        with tf.name_scope(name or '%s_pooling_along_time' % pooling_method):
            rep_mask_tiled = tf.tile(tf.expand_dims(rep_mask, 2), [1, 1, ivec])
            denominator = tf.reduce_sum(tf.cast(rep_mask, tf.int32), 1, keepdims=True)  # bs, 1
            denominator_tiled = tf.tile(denominator, [1, ivec])
            if pooling_method == 'mean':
                rep_tensor_zero = tf.where(rep_mask_tiled, rep_tensor, tf.zeros_like(rep_tensor))  # bs,sl,vec
                rep_sum = tf.reduce_sum(rep_tensor_zero, 1)  # bs, vec
                denominator = tf.where(tf.equal(denominator, 0), tf.ones_like(denominator), denominator)
                res = rep_sum / tf.cast(denominator, tf.float32)
            else:
                raise RuntimeError('no pooling method %s' % pooling_method)

            if keep_dim:
                res = tf.expand_dims(res, 1)
                if keep_len:
                    res = tf.tile(res, [1, sl, 1])
            return res

    def sequence_conditional_feature(self, rep_tensor, rep_mask):
        rep_tensor_pooling = self.pooling_along_time(rep_tensor, rep_mask, 'mean', True, True)
        rep_tensor_new = tf.concat([rep_tensor, rep_tensor_pooling, rep_tensor * rep_tensor_pooling], -1)
        return rep_tensor_new

    def Mask(self, inputs, seq_len, mode="add"):
        """Mask operation used in multi-head self attention

        Args:
            seq_len (object): sequence length of inputs.
            mode (str): mode of mask.

        Returns:
            object: tensors after masking.
        """

        if seq_len is None:
            return inputs
        else:
            mask = K.one_hot(indices=seq_len[:, 0], num_classes=K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, axis=1)

            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)

            if mode == "mul":
                return inputs * mask
            elif mode == "add":
                return inputs - (1 - mask) * 1e12

    def binary_entropy(probs, labels):
        with tf.name_scope('binary_entropy'):
            labels = tf.cast(labels, tf.float32)
            return - (labels * tf.log(tf.clip_by_value(probs, 1e-10, 1.0)) +
                      (1.0 - labels) * tf.log(tf.clip_by_value(1.0 - probs, 1e-10, 1.0)))

    def create_matrix(self, x, y):
        x=x>0.5
        y=y>0.5
        x_reshaped = tf.squeeze(x, axis=-1)
        y_reshaped = tf.squeeze(y, axis=-1)

        # Reshape x and y to have shape (None, n, 1) and (None, 1, n) respectively for broadcasting
        x_reshaped = tf.expand_dims(x_reshaped, axis=-1)
        y_reshaped = tf.expand_dims(y_reshaped, axis=-2)
        
        # Use broadcasting and element-wise comparison to create the matrix M
        M = tf.cast(tf.logical_and(x_reshaped, y_reshaped), tf.float32)

        return M

    def call(self, QKVs):
        """Core logic of multi-head self attention.

        Args:
            QKVs (list): inputs of multi-head self attention i.e. query, key and value.

        Returns:
            object: ouput tensors.
        """

        if len(QKVs) == 3:
            Q_seq, K_seq, V_seq = QKVs
            Q_len, V_len = None, None
        elif len(QKVs) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = QKVs
        
        rep_mask = tf.ones_like(Q_seq[:, :, -1], dtype=tf.bool)
        rss = self.sequence_conditional_feature(Q_seq, rep_mask)

        # K policy
        z_k = self.ln_1k(rss)
        z_k = self.ln_2k(z_k)

        #Q policy
        z_q = self.ln_1q(rss)
        z_q = self.ln_2q(z_q)

        M = self.create_matrix(z_k, z_q)
        self.mask = M

        rewards_raw = - tf.math.log(z_k)*tf.math.log(z_q)

        cost_rl = - tf.reduce_mean(rewards_raw * tf.reduce_sum(z_k, 1) + rewards_raw * tf.reduce_sum(z_q, 1), name='cost_rl')
        self.rl_loss = cost_rl

        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(
            Q_seq, shape=(-1, K.shape(Q_seq)[1], self.multiheads, self.head_dim)
        )
        Q_seq = K.permute_dimensions(Q_seq, pattern=(0, 2, 1, 3))

        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(
            K_seq, shape=(-1, K.shape(K_seq)[1], self.multiheads, self.head_dim)
        )
        K_seq = K.permute_dimensions(K_seq, pattern=(0, 2, 1, 3))

        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(
            V_seq, shape=(-1, K.shape(V_seq)[1], self.multiheads, self.head_dim)
        )
        V_seq = K.permute_dimensions(V_seq, pattern=(0, 2, 1, 3))

        A = einsum("abij, abkj -> abik", Q_seq, K_seq) / K.sqrt(
            K.cast(self.head_dim, dtype="float32")
        )
        print(A.shape)
        exit()
        A = tf.matmul(M[:, tf.newaxis, :, :], A)

        A = K.permute_dimensions(
            A, pattern=(0, 3, 2, 1)
        )  # A.shape=[batch_size,K_sequence_length,Q_sequence_length,self.multiheads]

        A = self.Mask(A, V_len, "add")
        A = K.permute_dimensions(A, pattern=(0, 3, 2, 1))

        if self.mask_right:
            ones = K.ones_like(A[:1, :1])
            lower_triangular = K.tf.matrix_band_part(ones, num_lower=-1, num_upper=0)
            mask = (ones - lower_triangular) * 1e12
            A = A - mask
        A = K.softmax(A)

        O_seq = einsum("abij, abjk -> abik", A, V_seq)
        O_seq = K.permute_dimensions(O_seq, pattern=(0, 2, 1, 3))

        O_seq = K.reshape(O_seq, shape=(-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, "mul")

        return O_seq

    def get_config(self):
        """add multiheads, multiheads and mask_right into layer config.

        Returns:
            dict: config of SelfAttention layer.
        """
        config = super(SelfAttention, self).get_config()
        config.update(
            {
                "multiheads": self.multiheads,
                "head_dim": self.head_dim,
                "mask_right": self.mask_right,
            }
        )
        return config


def PersonalizedAttentivePooling(dim1, dim2, dim3, seed=0):
    """Soft alignment attention implement.

    Attributes:
        dim1 (int): first dimention of value shape.
        dim2 (int): second dimention of value shape.
        dim3 (int): shape of query

    Returns:
        object: weighted summary of inputs value.
    """
    vecs_input = keras.Input(shape=(dim1, dim2), dtype="float32")
    query_input = keras.Input(shape=(dim3,), dtype="float32")

    user_vecs = layers.Dropout(0.2)(vecs_input)
    user_att = layers.Dense(
        dim3,
        activation="tanh",
        kernel_initializer=keras.initializers.glorot_uniform(seed=seed),
        bias_initializer=keras.initializers.Zeros(),
    )(user_vecs)
    user_att2 = layers.Dot(axes=-1)([query_input, user_att])
    user_att2 = layers.Activation("softmax")(user_att2)
    user_vec = layers.Dot((1, 1))([user_vecs, user_att2])

    model = keras.Model([vecs_input, query_input], user_vec)
    return model

class ComputeMasking(layers.Layer):
    """Compute if inputs contains zero value.

    Returns:
        bool tensor: True for values not equal to zero.
    """

    def __init__(self, **kwargs):
        super(ComputeMasking, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        mask = K.not_equal(inputs, 0)
        return K.cast(mask, K.floatx())

    def compute_output_shape(self, input_shape):
        return input_shape


class OverwriteMasking(layers.Layer):
    """Set values at spasific positions to zero.

    Args:
        inputs (list): value tensor and mask tensor.

    Returns:
        object: tensor after setting values to zero.
    """

    def __init__(self, **kwargs):
        super(OverwriteMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        super(OverwriteMasking, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs[0] * K.expand_dims(inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]
