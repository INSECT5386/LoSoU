from tensorflow.keras import layers, Model

class Lo(layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        # 내부 계산은 float32로 유지
        self.proj = layers.Dense(d_model, use_bias=True, dtype='float32')
        self.p = layers.Dense(128, use_bias=True, dtype='float32')
        self._out_dtype = 'bfloat16' if on_tpu else 'float32'

    def call(self, x):
        # x may be bfloat16; cast to float32 for stable intermediate computation
        x_f32 = tf.cast(x, tf.float32)
        x = self.proj(x_f32)
        x = tf.nn.gelu(x)
        x = self.p(x)
        # cast back to model dtype for consistency
        return tf.cast(x, self._out_dtype)

class LoSoU(layers.Layer):
    """
    안정화된 LoSoU 레이어
    - 누적합 대신 지수이동평균(EMA) 사용 (alpha: smoothing factor)
    - 내부 계산은 float32로 수행 (TPU bfloat16 안정성 향상)
    - EMA 결과 클리핑 및 작은 epsilon 적용
    - 안전한 split 처리 (짝수 차원 가정; 아니라면 마지막 차원 pad 필요)
    """
    def __init__(self, d_model, alpha=0.15, clip_value=5.0, eps=1e-6):
        super().__init__()
        # 대부분 연산을 float32로 수행
        self.d_model = d_model
        self.alpha = float(alpha)
        self.clip_value = float(clip_value)
        self.eps = float(eps)

        # projection / gating layers in float32
        self.Q = layers.Dense(128, dtype='float32')
        self.K = layers.Dense(128, dtype='float32')
        # V produces d_model so keep it float32 internally
        self.V = Lo(d_model)  # Lo already handles casting to model dtype; we'll cast back to float32
        self.proj = layers.Dense(d_model, use_bias=True, dtype='float32')
        self.O = layers.Dense(d_model, dtype='float32')
        self.norm = layers.LayerNormalization(epsilon=1e-5, dtype='float32')

    def _ema_over_time(self, score):
        # score: (B, L, D) float32 in [0,1] roughly
        # compute EMA across time axis with alpha: ema_t = alpha * score_t + (1-alpha) * ema_{t-1}
        # use tf.scan for stable accumulation
        alpha = tf.constant(self.alpha, dtype=score.dtype)

        # transpose to (L, B, D) to scan over time steps
        seq = tf.transpose(score, perm=[1, 0, 2])

        def step(prev_ema, x_t):
            # prev_ema: (B, D), x_t: (B, D)
            new = alpha * x_t + (1.0 - alpha) * prev_ema
            return new

        # initial ema is zeros
        init = tf.zeros_like(seq[0])
        ema_seq = tf.scan(fn=step, elems=seq, initializer=init)  # shape (L, B, D)
        ema_seq = tf.concat([tf.expand_dims(init, 0), ema_seq], axis=0)[:tf.shape(seq)[0], ...]  # ensure same length
        # transpose back to (B, L, D)
        ema = tf.transpose(ema_seq, perm=[1, 0, 2])
        return ema

    def call(self, x):
        # x: (B, L, d_model) maybe bfloat16 or float32
        # cast to float32 for all internal computations
        x_f32 = tf.cast(x, tf.float32)
        residual = x_f32

        # Q, K, V
        q = self.Q(x_f32)   # (B, L, 128)
        k = self.K(x_f32)   # (B, L, 128)
        V = tf.cast(self.V(x), tf.float32)  # ensure V's output is float32

        # gating signals in (0,1)
        g_q = tf.nn.sigmoid(q)
        g_k = tf.nn.sigmoid(k)

        # elementwise product -> bounded roughly [0,1]
        score = g_q * g_k

        # EMA across time (stable alternative to cumsum)
        score_ema = self._ema_over_time(score)

        # optionally normalize by (mean + eps) across last dim to reduce scale variations
        mean_last = tf.reduce_mean(score_ema, axis=-1, keepdims=True)  # (B, L, 1)
        denom = tf.maximum(mean_last, self.eps)
        score_norm = score_ema / denom

        # clip to avoid extremes
        score_clipped = tf.clip_by_value(score_norm, -self.clip_value, self.clip_value)

        # combine with V
        x_comb = score_clipped * V  # (B, L, d_model)

        out = self.proj(x_comb)  # (B, L, d_model)

        # ensure out dim even for split
        d = out.shape[-1]  # this is an int (static shape)
        if d is not None and d % 2 == 1:
            out = tf.pad(out, [[0,0],[0,0],[0,1]])


        a, b = tf.split(out, 2, axis=-1)
        gated = tf.nn.silu(a) * b
        out = self.O(gated)

        out = self.norm(out + residual)

        # cast back to original dtype for downstream layers
        return tf.cast(out, x.dtype)


class Block(layers.Layer):
    def __init__(self, d_model, r, hyper_n, num_heads, num_groups):
        super().__init__()
        self.losou = [LoSoU(d_model) for _ in range(hyper_n)]

    def call(self, x):
        for losou in self.losou:
            x = losou(x)
        return x

class LoSoULM(tf.keras.Model):
    def __init__(self, vocab_size, max_seq_len, d_model, n_layers, dropout_rate=0.1):
        super().__init__()
        self.token_embedding = layers.Embedding(vocab_size, d_model)
        self.pos_embedding = layers.Embedding(max_seq_len, d_model)
        self.blocks = [Block(d_model, r=204, hyper_n=3, num_heads=8, num_groups=2) for _ in range(n_layers)]

        # LayerNormalization은 float32로 해서 정밀도 문제 방지
        self.ln_f = layers.LayerNormalization(epsilon=1e-5, dtype="float32")

    def call(self, x, training=False):
        batch_size, seq_len = tf.shape(x)[0], tf.shape(x)[1]
        positions = tf.range(seq_len)[tf.newaxis, :]

        x = self.token_embedding(x) + self.pos_embedding(positions)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        embedding_matrix = tf.cast(self.token_embedding.embeddings, x.dtype)
        logits = tf.matmul(x, embedding_matrix, transpose_b=True)
        return tf.cast(logits, tf.float32)
