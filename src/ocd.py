import torch
import torch.nn.functional as F


def ocd(ref):
    table = [list(range(len(ref)))]
    def step(h):
        i = len(table)
        if h != None:
            row = [i]
            for j, r in enumerate(ref[:-1]):
                j += 1
                val = min(table[i-1][j] + 1, \
                        row[j-1] + 1, table[i-1][j-1] + (h != r))
                row.append(val)
            table.append(row)
        min_d = min(table[-1])
        cands_idx = [idx for idx, d in enumerate(table[-1]) if d == min_d]
        cands = [ref[i] for i in cands_idx]
        return cands, min_d

    return step


def ocd_loss(out_probs, samples, labels, temp=1e-8):
    temp = max(temp, 1e-8) # to prevent division by zero
    vocab_size = out_probs.size(-1)
    loss = 0
    batch_size = len(labels)

    for b, (sample, label) in enumerate(zip(samples, labels)):
        sample = sample.tolist()
        try:
            len_sample = sample.index(1) + 1 # length of sample is where the first <eos> appears
        except ValueError:
            len_sample = len(sample)
        sample = sample[:len_sample]
        label = label[label != 0].tolist() # exclude pad token
        q_val = out_probs.new_zeros((len_sample, vocab_size))
        play = ocd(label)
        for t, char in enumerate(['<sos>'] + sample[:-1]):
            cands, m = play(char)
            cands = torch.LongTensor(cands)
            q_val[t] = -(m + 1)
            q_val[t,cands] = -m

        loss += - (F.softmax(q_val / temp, dim=-1) * F.log_softmax(out_probs[b,:len_sample,:]))\
                .sum(dim=-1).mean()

    loss /= batch_size
    return loss

