from losou import LoSoU
from tensorflow import keras
import sentencepiece as spm
import os, json, numpy as np, tensorflow as tf

tf.get_logger().setLevel("ERROR")
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# TPU 초기화
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("✅ TPU 초기화 완료:", resolver.cluster_spec().as_dict())
    on_tpu = True
except Exception as e:
    print("⚠️ TPU 미사용, GPU/CPU로 진행:", e)
    strategy = tf.distribute.get_strategy()
    on_tpu = False

# Mixed precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy("mixed_bfloat16" if on_tpu else "float32")
mixed_precision.set_global_policy(policy)
print("✅ Mixed precision:", policy)

# =======================
# 1) 파일 다운로드
# =======================
def download_file(url, save_path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    print(f"✅ {save_path} 저장됨")

DATA_PATH = "converted.jsonl"
TOKENIZER_PATH = "ko_unigram.model"

if not os.path.exists(DATA_PATH):
    download_file(
        "https://huggingface.co/datasets/Yuchan5386/SFT/resolve/main/data_shuffled_1.jsonl?download=true",
        DATA_PATH
    )

if not os.path.exists(TOKENIZER_PATH):
    download_file(
        "https://huggingface.co/Yuchan5386/inlam-100m/resolve/main/ko_unigram.model?download=true",
        TOKENIZER_PATH
    )

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)

pad_id = sp.piece_to_id("<pad>") if sp.piece_to_id("<pad>") != -1 else 0
start_id = sp.piece_to_id("<start>")
sep_id = sp.piece_to_id("<sep>")
end_id = sp.piece_to_id("<end>")
unk_id = sp.piece_to_id("<unk>")
vocab_size = sp.get_piece_size()
print(f"✅ Vocabulary size: {vocab_size}")

max_len = 200
batch_size = 128

def text_to_ids(text):
    return sp.encode(text, out_type=int)
def ids_to_text(ids):
    return sp.decode(ids)

# (생략 가능) 데이터 로더는 원본과 동일

def jsonl_stream(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            conversations = data.get("conversations", [])
            for i in range(0, len(conversations) - 1, 2):
                human_msg = conversations[i]
                gpt_msg   = conversations[i + 1]
                if human_msg.get("from") != "human" or gpt_msg.get("from") != "gpt":
                    continue
                prompt   = human_msg.get("value", "").strip()
                response = gpt_msg.get("value", "").strip()
                full = f"<start> {prompt} <sep> {response} <end>"
                if "<sep>" not in full:
                    continue
                sep_index  = full.index("<sep>")
                input_text = full[:sep_index + len("<sep>")].strip()
                target_text = full[sep_index + len("<sep>"):].strip()

                input_ids  = text_to_ids(input_text)
                target_ids = text_to_ids(target_text + " <end>")

                available_len = max_len - len(input_ids)
                if available_len <= 0:
                    input_ids = input_ids[-max_len:]
                    target_ids = []
                    target_mask = [0] * len(input_ids)
                else:
                    target_ids = target_ids[:available_len]
                    target_mask = [0] * len(input_ids) + [1] * len(target_ids)

                full_input = input_ids + target_ids
                pad_len = max_len - len(full_input)
                full_input += [pad_id] * pad_len
                target_mask += [0] * pad_len

                target_seq = full_input[1:] + [end_id]
                target_seq = target_seq[:max_len]

                masked_target = [
                    t if m == 1 else pad_id
                    for t, m in zip(target_seq, target_mask)
                ]

                yield (
                    tf.convert_to_tensor(full_input, dtype=tf.int32),
                    tf.convert_to_tensor(masked_target, dtype=tf.int32)
                )

dataset = tf.data.Dataset.from_generator(
    lambda: jsonl_stream(DATA_PATH),
    output_signature=(
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
    ),
)
dataset = dataset.shuffle(1000, seed=SEED).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

with strategy.scope():
    dist_dataset = strategy.experimental_distribute_dataset(dataset)



def smoothed_loss_keras(y_true, y_pred, eps=0.1):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    vocab = tf.shape(y_pred)[-1]
    y_true_oh = tf.one_hot(y_true, depth=vocab, dtype=tf.float32)
    y_true_ls = (1.0 - eps) * y_true_oh + eps / tf.cast(vocab, tf.float32)
    log_probs = tf.nn.log_softmax(y_pred, axis=-1)
    per_tok = -tf.reduce_sum(y_true_ls * log_probs, axis=-1) * mask
    return tf.reduce_sum(per_tok) / (tf.reduce_sum(mask) + 1e-8)

def masked_accuracy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    pred_id = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    acc = tf.cast(tf.equal(y_true, pred_id), tf.float32) * mask
    return tf.reduce_sum(acc) / (tf.reduce_sum(mask) + 1e-8)

# =======================
# 모델 생성 & 학습
# =======================
with strategy.scope():
    model = Sequen(vocab_size, max_seq_len=max_len, d_model=384, n_layers=12, dropout_rate=0.1)
    dummy_input = tf.zeros((batch_size, max_len), dtype=tf.int32)
    _ = model(dummy_input, training=False)
    model.summary()

    # Adam epsilon 키워서 안정화
    optimizer = tf.keras.optimizers.Adam(3e-4, beta_1=0.9, beta_2=0.95, epsilon=1e-6, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=smoothed_loss_keras, metrics=[masked_accuracy])
    history = model.fit(dist_dataset, epochs=1, verbose=1)

# =======================
# 가중치 저장
# =======================
model.save_weights("Sequen.weights.h5")
print("✅ 모델 가중치 저장 완료!")

# =======================
@tf.function(input_signature=[
    tf.TensorSpec(shape=(1, None), dtype=tf.int32),  # input_ids
    tf.TensorSpec(shape=(vocab_size,), dtype=tf.int32),  # token_counts
    tf.TensorSpec(shape=(), dtype=tf.int32),  # current_length
    tf.TensorSpec(shape=(), dtype=tf.float32),  # temperature
    tf.TensorSpec(shape=(), dtype=tf.float32),  # repetition_penalty
    tf.TensorSpec(shape=(), dtype=tf.float32),  # top_p
    tf.TensorSpec(shape=(), dtype=tf.int32),  # top_k
    tf.TensorSpec(shape=(), dtype=tf.int32),  # min_len
    tf.TensorSpec(shape=(), dtype=tf.int32),  # step
])
def generate_step(input_ids, token_counts, current_length, temperature, repetition_penalty, top_p, top_k, min_len, step):
    pad_len = max_len - tf.shape(input_ids)[1]
    input_padded = tf.pad(input_ids, [[0,0],[0,pad_len]], constant_values=pad_id)
    logits = model(input_padded, training=False)
    next_logits = logits[0, current_length - 1]

    penalty = tf.pow(repetition_penalty, tf.cast(token_counts, tf.float32))
    next_logits = next_logits / penalty

    # 최소 길이와 pad 마스킹
    if current_length < min_len:
        next_logits = tf.tensor_scatter_nd_update(next_logits, [[end_id]], [-1e9])
    next_logits = tf.tensor_scatter_nd_update(next_logits, [[pad_id]], [-1e9])

    # top-k 필터링
    if top_k > 0:
        kth_val = tf.math.top_k(next_logits, k=top_k).values[-1]
        mask = next_logits < kth_val
        next_logits = tf.where(mask, -1e9, next_logits)

    # top-p (nucleus) 필터링 + temperature
    next_logits = next_logits / temperature
    probs = tf.nn.softmax(next_logits)
    sorted_probs, sorted_idx = tf.math.top_k(probs, k=vocab_size)
    cum_probs = tf.cumsum(sorted_probs)
    cutoff_mask = cum_probs <= top_p
    cutoff_idx = tf.reduce_sum(tf.cast(cutoff_mask, tf.int32)) + 1
    cutoff_idx = tf.minimum(cutoff_idx, vocab_size)
    filtered_idx = sorted_idx[:cutoff_idx]
    filtered_probs = sorted_probs[:cutoff_idx]
    filtered_probs = filtered_probs / tf.reduce_sum(filtered_probs)

    # 50%는 argmax, 50%는 샘플링 (버그 수정)
    rand_val = tf.random.uniform([], 0, 1)
    def sample():
        sampled_id = tf.random.categorical(tf.math.log([filtered_probs]), 1)[0,0]
        return filtered_idx[sampled_id]
    def argmax():
        return filtered_idx[tf.argmax(filtered_probs)]
    sampled_id = tf.cond(rand_val < 0.5, argmax, sample)
    sampled_id = tf.cast(sampled_id, tf.int32)

    # token_counts 업데이트
    token_counts = tf.tensor_scatter_nd_add(token_counts, [[sampled_id]], [1])
    return sampled_id, token_counts


# =====================
# 스트리밍 생성기 (CPU 최적화 버전)
# =====================
def generate_text_streaming(model, prompt, max_len=115, max_gen=100,
                            temperature=0.75, min_len=20,
                            repetition_penalty=1.2, top_p=0.9, top_k=50):
    model_input = text_to_ids(f"<start> {prompt} <sep>")
    model_input = model_input[:max_len]
    generated = list(model_input)
    start_output_idx = len(model_input)

    # TF 변수로 토큰 카운트 관리
    token_counts_np = np.zeros(vocab_size, dtype=np.int32)
    for t in generated:
        token_counts_np[t] += 1
    token_counts = tf.Variable(token_counts_np, dtype=tf.int32)

    prev_decoded = ""

    for step in range(max_gen):
        input_tensor = tf.expand_dims(generated, axis=0)  # [1, seq_len]

        sampled_id, token_counts = generate_step(
            input_tensor,
            token_counts,
            tf.constant(len(generated), dtype=tf.int32),
            tf.constant(temperature, dtype=tf.float32),
            tf.constant(repetition_penalty, dtype=tf.float32),
            tf.constant(top_p, dtype=tf.float32),
            tf.constant(top_k, dtype=tf.int32),
            tf.constant(min_len, dtype=tf.int32),
            tf.constant(step, dtype=tf.int32)
        )

        sampled_id = int(sampled_id.numpy())
        generated.append(sampled_id)

        # 디코딩은 출력 시점에만
        if len(generated) > start_output_idx:
            decoded_full = sp.decode(generated[start_output_idx:])
            decoded_full = decoded_full.replace("▁", " ").strip()
            for t in ["<start>", "<sep>", "<end>"]:
                decoded_full = decoded_full.replace(t, "")
            decoded_full = decoded_full.lstrip(",!?.는은 ")

            new_output = decoded_full[len(prev_decoded):]
            if new_output:
                yield new_output
                prev_decoded = decoded_full

            # 종료 조건
            if len(generated) >= min_len and (sampled_id == end_id or decoded_full.endswith(('.', '!', '?'))):
                break


for token in generate_text_streaming(
    model, '안녕하세요',
    max_len=max_len,
    max_gen=115,
    temperature=0.8,
    min_len=10,
    repetition_penalty=1.1,
    top_p=0.9,
    top_k=32
):
    print(token, end="", flush=True)
