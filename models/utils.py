def kl_divergence(p, q):
    # Assuming p and q are already normalized (softmax applied)
    return (p * (p / q).log()).sum(dim=-1)

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
