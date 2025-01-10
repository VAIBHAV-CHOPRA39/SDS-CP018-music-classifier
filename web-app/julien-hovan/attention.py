import tensorflow as tf
from tensorflow.keras import layers

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension needs to be divisible by number of heads"
        
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=self.head_dim
        )
        self.dropout = layers.Dropout(dropout_rate)
        self.layernorm = layers.LayerNormalization()
        
    def call(self, inputs, training=None):
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs
        )
        attention_output = self.dropout(attention_output, training=training)
        return self.layernorm(inputs + attention_output) 
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout.rate,
        })
        return config