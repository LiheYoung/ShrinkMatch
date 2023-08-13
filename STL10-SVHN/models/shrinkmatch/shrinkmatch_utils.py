import torch
import torch.nn.functional as F
from train_utils import ce_loss


class Get_Scalar:
    def __init__(self, value):
        self.value = value
        
    def get_value(self, iter):
        return self.value
    
    def __call__(self, iter):
        return self.value


def consistency_loss_with_shrink(logits_s, logits_w, logits_aux_s, p_cutoff=0.0, use_da=False, p_model=None):
    logits_w = logits_w.detach()
    
    pseudo_label = torch.softmax(logits_w, dim=-1)
    
    if use_da:
        if p_model == None:
            p_model = torch.mean(pseudo_label.detach(), dim=0)
        else:
            p_model = p_model * 0.999 + torch.mean(pseudo_label.detach(), dim=0) * 0.001
        pseudo_label = pseudo_label * (1.0 / p_model)
        pseudo_label = (pseudo_label / pseudo_label.sum(dim=-1, keepdim=True))
    
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(p_cutoff).float()
    
    masked_loss = ce_loss(logits_s, max_idx, True, reduction='none') * mask
    
    sorted_logits_w, sorted_idx = logits_w.topk(logits_w.shape[-1], dim=-1, sorted=True)
    sorted_logits_s = logits_aux_s.gather(dim=-1, index=sorted_idx)
    
    B, C = logits_s.shape
    removed_class_idx = []
    uncertainty_avg = []
    shrink_loss = 0
    
    if mask.mean().item() == 1: # no uncertain samples to shrink
        shrink_loss = 0
    else:
        for b in range(B):
            if max_probs[b] >= p_cutoff: # skip certain samples
                continue
            uncertainty_avg.append(max_probs[b].item())
            
            # iteratively remove classes to enhance confidence until satisfying the confidence threshold
            for c in range(2, C):
                # new confidence in the shrunk class space (classes ranging from 1 ~ (c-1) are removed)
                sub_conf = torch.cat([sorted_logits_w[b, :1], sorted_logits_w[b, c:]], dim=0).softmax(dim=0)[0]
                
                # break either when satifying the threshold or traversing to the final class (with smallest value)
                if (sub_conf >= p_cutoff) or (c == C - 1):
                    sub_logits_s = torch.cat([sorted_logits_s[b, :1], sorted_logits_s[b, c:]], dim=0)
                    
                    shrink_loss_cur = ce_loss(sub_logits_s.unsqueeze(0), torch.zeros(1).long().cuda(), True, reduction='none')[0]
                    # for our loss reweighting principle 1
                    shrink_loss_cur *= max_probs[b] * ((sub_conf >= p_cutoff))
                    
                    shrink_loss += shrink_loss_cur
                    
                    removed_class_idx.append(c)
                    
                    break
    
    return masked_loss.mean(), mask.mean(), shrink_loss / B, \
        sum(removed_class_idx) / (len(removed_class_idx) + 1e-5), sum(uncertainty_avg) / (len(uncertainty_avg) + 1e-5), p_model