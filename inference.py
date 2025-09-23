from losou import LoSoULM

# =====================
# 모델/토크나이저 다운로드
# =====================
os.environ["HF_HOME"] = "/tmp/hf_cache"
hf_token = os.getenv("HF_TOKEN")

CHAT_MODEL_PATH = hf_hub_download(
    repo_id="Yuchan5386/LoSoU-35m-instruct",
    filename="LoSoULM.weights.h5",
    repo_type="model",
    token=hf_token
)
CHAT_TOKENIZER_PATH = hf_hub_download(
    repo_id="Yuchan5386/LoSoU-35m-instruct",
    filename="ko_unigram.model",
    repo_type="model",
    token=hf_token
)


print(CHAT_MODEL_PATH)
sp = spm.SentencePieceProcessor()  
sp.load(CHAT_TOKENIZER_PATH) 
pad_id = sp.piece_to_id("<pad>") or 0
start_id = sp.piece_to_id("<start>") or 1
end_id = sp.piece_to_id("<end>") or 2
unk_id = sp.piece_to_id("<unk>") or 3
sep_id = sp.piece_to_id("<sep>")
vocab_size = sp.get_piece_size()
max_len = 200

def text_to_ids(text):
    return sp.encode(text, out_type=int)

def ids_to_text(ids):
    return sp.decode(ids)

model = LoSoULM(vocab_size, max_seq_len=max_len, d_model=384, n_layers=12, dropout_rate=0.1)
dummy_input = tf.zeros((1, max_len), dtype=tf.int32)  # 배치1, 시퀀스길이 max_len  
_ = model(dummy_input)  # 모델이 빌드됨  
model.load_weights(CHAT_MODEL_PATH)  
print("모델 가중치 로드 완료!")  

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
