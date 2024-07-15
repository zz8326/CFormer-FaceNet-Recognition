import tensorflow as tf
from tensorflow.keras import layers, activations
import numpy as np
class conv_block(layers.Layer):
    """
    Conv Block: 實現Cformer 捲積模塊
    ↑ →------------------------------------------------------→
    ↑                                                        ↓
    Input →→ DW layer →→ Norm →→ Linear →→ gleu →→ Linear →→ ADD →→ output
    𝑥𝑖 + 1=𝑥𝑖+ 𝑃𝑤𝐺𝐸𝐿𝑈( 𝑃 𝑤 ( 𝐿 𝑁( 𝐷 𝑤 (𝑥𝑖) ) ) )
    修改: 原論文採用LayerNorm →→BatchNorm
    """
    def __init__(self, dim, kernel_size, name = None):
        super(conv_block, self).__init__(name=name)
        self.spatial_dwconv = tf.keras.layers.Conv2D(dim, kernel_size=kernel_size, strides=1,
                                                     padding='same', groups=dim, use_bias=True) # 深度捲積
        self.norm = layers.BatchNormalization(axis=-1)
        self.act = activations.gelu
        self.pwconv1 = layers.Conv2D(4 * dim, kernel_size=(1,1), padding='same')
        self.pwconv2 = layers.Conv2D(dim, kernel_size=(1, 1), padding='same')
    def call(self, x):
        spatial_output = self.spatial_dwconv(x)
        norm = self.norm(spatial_output)
        layer_channel_output = self.pwconv1(norm)
        layer_act_output = self.act(layer_channel_output)
        layer_channel_output = self.pwconv2(layer_act_output)
        output = x + layer_channel_output
        return output

class GOTA_block_part1(layers.Layer):
    """
    GDTA 上半段
    功能: 將特徵圖分組，獨立進行捲積計算，在利用殘差的方式保留訊息
    self.num_groups: 通道分成S組
    self.input_dim: 輸入維度
    self.output_dim:　輸出維度
    self.conv_layers：　創建3*3 kernel的捲積列表
    self.concat_layer : 合併層
    """
    def __init__(self, dim, num_groups, name= None):
        super(GOTA_block_part1, self).__init__(name=name)
        assert dim % num_groups == 0, 'num_groups must divide dim'
        self.num_groups = num_groups
        self.input_dim = dim
        self.output_dim = dim // num_groups
        self.conv_layers = [layers.Conv2D(dim // num_groups, kernel_size=(3, 3), padding='same', groups=dim // num_groups) for _ in range(num_groups)]
        self.concat_layer = layers.Concatenate()
    def call(self, x):
        groups_out = []
        for i in range(self.num_groups):
            if i == 0:
                yi = x[:, :, :, :self.output_dim]
            else:
                x_i = x[:, :, :, i * self.output_dim: (i + 1) * self.output_dim]
                K_i = self.conv_layers[i](x_i)
                if i == 1:
                    yi = K_i
                else:
                    yi = K_i + groups_out[-1]
            groups_out.append(yi)
        output = self.concat_layer(groups_out)
        return output

class Attention(layers.Layer):
    """
    GDTA下半段: Attention method
    主要改進的地方:
    Transformer 對大的挑戰在於點積計算時會造成計算複雜度大幅度上升
    GDTA使用 transpose q @ k的方式在通道進行MSA而非空間維度上
    進入點積運算前使用L2泛數增強穩定性
    """
    def __init__(self, dim, num_heads, qkv_bias = True, dropout_rate = 0.0, name = None):
        super(Attention, self).__init__(name=name)
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)
    def call(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, shape=(-1, N, 3, self.num_heads, C//self.num_heads))
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = tf.nn.l2_normalize(q, axis=-1)
        k = tf.nn.l2_normalize(k, axis=-1)
        q = q*self.scale
        #k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn_scores = tf.matmul(q, k, transpose_a=True)
        attn_weight = tf.nn.softmax(attn_scores, axis=-1)
        output = tf.matmul(v,attn_weight)
        output = tf.transpose(output, perm=(0, 2, 1, 3))
        output= tf.reshape(output, shape=(-1, N, C))
        output = self.proj(output)
        output = self.dropout(output)
        return output
class MLP(layers.Layer):
    """
    MLP層實現
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(MLP,self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = layers.Conv2D(hidden_features, kernel_size=(1, 1), padding='same')
        self.act = tf.keras.activations.gelu
        # hidden_features=256, out_features=64
        self.fc2 = layers.Conv2D(out_features, kernel_size=(1, 1), padding='same')
        self.drop = tf.keras.layers.Dropout(drop)
    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class GDTA_block(layers.Layer):
    """
    整體GDTA的架構，結合上半部的分離捲積計算以及下半部的Attention運算
    """
    def __init__(self, dim, num_heads, num_groups, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., name = None):
        super(GDTA_block, self).__init__(name=name)
        self.dim = dim
        self.num_heads = num_heads
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.dwconv = GOTA_block_part1(dim = dim, num_groups= num_groups)
        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, dropout_rate=drop)
        mlp_hidden_dim = int(dim * mlp_ratio)  #
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    def call(self, x):
        _, H, W, C = x.shape
        dwconv = self.dwconv(x)
        norm1 = self.norm1(dwconv)
        norm1 = tf.reshape(norm1, shape=(-1, H*W, C))
        attn = self.attn(norm1)
        attn = tf.reshape(attn, shape=(-1, H, W, C))
        Merging = attn + dwconv
        Merging = self.norm2(Merging)
        mlp = self.mlp(Merging)
        output = x + mlp
        return output
class AddPosEmbed(layers.Layer):
    def __init__(self,img_len):
        super(AddPosEmbed, self).__init__()
        self.img_len = img_len

    def build(self, input_shape):
        self.pos_embed = self.add_weight(shape=[1,self.img_len,input_shape[-1]],
                                        initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None),trainable=True,dtype=tf.dtypes.float32)
    def call(self, x):
        return x+self.pos_embed

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'img_len': self.img_len,
        })
        return config
def get_cformer(embed_dims = [32, 64, 96, 128, 512], kernel_size = [3, 5, 7, 9],patch_num = [56, 28, 14, 7],
                depth=[3, 5, 5, 2],num_heads = [1, 2, 4, 8], num_groups = 4, qkv_bias = True,
                drop = 0.0, mlp_ratio = [4., 4., 4., 4.]):
    """
    建構模型，Arcface層用於Train.py
    """
    input = layers.Input(shape=(112, 112, 3))
    """stage 1"""
    #down sampling
    x = layers.Conv2D(embed_dims[0], kernel_size=(2, 2), strides=(2, 2), padding='same',
                      name='stage_1_downsampling_layer')(input)
    #x = layers.LayerNormalization(epsilon=1e-6)(x)
    # conv block
    for i in range(depth[0]):
        x = conv_block(embed_dims[0], kernel_size=(kernel_size[0], kernel_size[0]),
                       name=f'stage_1_convblock{i}')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    """stage 2"""
    # down sampling
    x = layers.Conv2D(embed_dims[1], kernel_size=(2, 2), strides=(2, 2), padding='same',
                      name='stage_2_downsampling_layer')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    #x = layers.LayerNormalization(epsilon=1e-6)(x)
    # conv block
    for i in range(depth[1]):
        x = conv_block(embed_dims[1], kernel_size=(kernel_size[1], kernel_size[1]),
                       name=f'stage_2_convblock{i}')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    #GDTA block
    x = tf.reshape(x , shape=(-1, x.shape[1]*x.shape[2], x.shape[3]))
    x = AddPosEmbed(img_len=patch_num[1]**2)(x)
    x = tf.reshape(x, shape=(-1, patch_num[1], patch_num[1], embed_dims[1]))
    x = GDTA_block(dim=embed_dims[1], num_heads=num_heads[0], num_groups=num_groups,
                   qkv_bias=qkv_bias, mlp_ratio=mlp_ratio[0], drop=drop, name='stage2_GDTA_block')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    """stage 3"""
    # down sampling
    x = layers.Conv2D(embed_dims[2], kernel_size=(2, 2), strides=(2, 2), padding='same',
                      name='stage_3_downsampling_layer')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    # x = layers.LayerNormalization(epsilon=1e-6)(x)
    # conv block
    for i in range(depth[2]):
        x = conv_block(embed_dims[2], kernel_size=(kernel_size[2], kernel_size[2]),
                       name=f'stage_3_convblock{i}')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    # GDTA block
    x = tf.reshape(x, shape=(-1, x.shape[1] * x.shape[2], x.shape[3]))
    x = AddPosEmbed(img_len=patch_num[2]**2)(x)
    x = tf.reshape(x, shape=(-1, patch_num[2], patch_num[2], embed_dims[2]))
    x = GDTA_block(dim=embed_dims[2], num_heads=num_heads[1], num_groups=num_groups,
                   qkv_bias=qkv_bias, mlp_ratio=mlp_ratio[1], drop=drop, name='stage3_GDTA_block')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    """stage 4"""
    # down sampling
    x = layers.Conv2D(embed_dims[3], kernel_size=(2, 2), strides=(2, 2), padding='same',
                      name='stage_4_downsampling_layer')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    # x = layers.LayerNormalization(epsilon=1e-6)(x)
    # conv block
    for i in range(depth[3]):
        x = conv_block(embed_dims[3], kernel_size=(kernel_size[3], kernel_size[3]),
                       name=f'stage_4_convblock{i}')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    # GDTA block
    x = tf.reshape(x, shape=(-1, x.shape[1] * x.shape[2], x.shape[3]))
    x = AddPosEmbed(img_len=patch_num[3]**2)(x)
    x = tf.reshape(x, shape=(-1, patch_num[3], patch_num[3], embed_dims[3]))
    x = GDTA_block(dim=embed_dims[3], num_heads=num_heads[2], num_groups=num_groups,
                   qkv_bias=qkv_bias, mlp_ratio=mlp_ratio[2], drop=drop, name='stage4_GDTA_block')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(embed_dims[4], kernel_size=(1, 1), strides=(1, 1),padding='valid', name='extract_layer')(x)
    x = layers.SeparableConv2D(embed_dims[4], kernel_size=(7,7), use_bias=False,
                               padding='valid', name = 'extract_sepconv2')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    out = layers.GlobalAveragePooling2D(name='Global_Average_Pooling')(x)
    model = tf.keras.models.Model(inputs=input, outputs=out)
    return model

