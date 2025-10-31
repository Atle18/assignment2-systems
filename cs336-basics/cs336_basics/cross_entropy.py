import torch

def cross_entropy(logits, target):
    '''
    logit: [N, C], target: [N]
    '''
    # Since the range of softmax(logit, dim=-1) is [0, 1], log(0) is -inf which is underflow.
    # Therefore, this naive method is not numerical stable. 
    # loss = -torch.log(softmax(logit, dim=-1))[torch.arange(logit.shape[0]), target]
    # return loss.mean()
    
    # Using log_sum_exp skill    
    m = logits.max(dim=-1, keepdim=True).values
    lse = m + torch.log(torch.sum(torch.exp(logits - m), dim=-1, keepdim=True))
    log_probs = logits - lse
    loss = -log_probs[torch.arange(logits.size(0)), target]
    return loss.mean()