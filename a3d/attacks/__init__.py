"""
Helper function to estimate a custom epsilon for each sample in a batch.
"""
def get_energy_ratio(audio, grads, desired_snr):
    S_signal = (audio*audio).sum(1) / audio.size(1) # Batched dot-product
    S_noise = (grads*grads).sum(1) / grads.size(1)  # Batched dot-product
    ratio = (S_signal / (S_noise + 1e-8)) * (10 ** (-desired_snr / 10))

    return ratio.unsqueeze(1)