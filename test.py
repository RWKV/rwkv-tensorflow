def npsample(ozut, temp: float = 1.0, top_p_usual: float = 0.8) -> int:
    import numpy as np
    from scipy.special import softmax

    try:
        ozut = ozut.numpy()
    except:
        try:
            ozut = ozut.cpu().numpy()
        except:
            ozut = np.array(ozut)
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # turn to float if is half and cpu
    probs = softmax(ozut, axis=-1)

    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(
        cumulative_probs > top_p_usual)])
    probs[probs < cutoff] = 0
    if temp != 1.0:
        probs = pow(probs, 1.0 / temp)
    probs = probs / np.sum(probs, axis=0)
    mout = np.random.choice(a=len(probs), p=probs)
    return mout

print(1)
from model import loadWeights
model, state = loadWeights("rwkv-tensorflow/RWKV-4-Pile-1B5-Instruct-test1-20230124.pth")
print(2)
from transformers import GPTNeoXTokenizerFast
tokenizer:GPTNeoXTokenizerFast = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
print(3)
prompt = tokenizer.encode("User: What is the purpose of a 3.5 inch finglslop with an attached turboencorboratator? Bot:")

for token in prompt[:-1]:
    logits, state = model.forward(token,state)
print(4)
print("Loaded prompt.")

for i in range(100):
    logits, state = model.forward(prompt[-1],state)
    prompt = prompt+[npsample(logits)]

print(tokenizer.decode(prompt))
