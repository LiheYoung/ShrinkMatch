import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class BackBone(nn.Module):
    def __init__(self, base_encoder, num_classes):
        super(BackBone, self).__init__()
        self.net = base_encoder
        self.fc = nn.Linear(self.net.out_dim, num_classes)
        self.head_aux = nn.Sequential(
            nn.Linear(self.net.out_dim, self.net.out_dim // 4),
            nn.BatchNorm1d(self.net.out_dim // 4),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            
            nn.Linear(self.net.out_dim // 4, self.net.out_dim // 4),
            nn.BatchNorm1d(self.net.out_dim // 4),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            
            nn.Linear(self.net.out_dim // 4, num_classes)
        )

    def forward(self, x, bs=None):
        x = self.net(x)
        logits = self.fc(x)
        
        if self.training:
            logits_aux = self.head_aux(x[bs:])
            return logits, logits_aux
        
        return logits


class ShrinkMatch(nn.Module):
    def __init__(self, base_encoder, num_classes=10,  momentum=0.999, args=None):
        super(ShrinkMatch, self).__init__()
        self.m = momentum
        self.num_classes = num_classes
        self.encoder_q = BackBone(base_encoder, num_classes)
        self.ema = copy.deepcopy(self.encoder_q)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.ema.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        if args.DA:
            self.DA_len = 32
            self.register_buffer("DA_queue", torch.zeros(self.DA_len, num_classes, dtype=torch.float))
            self.register_buffer("DA_ptr", torch.zeros(1, dtype=torch.long))
    
    def momentum_update_ema(self):
        for param_train, param_eval in zip(self.encoder_q.parameters(), self.ema.parameters()):
            param_eval.copy_(param_eval * self.m + param_train.detach() * (1-self.m))
        for buffer_train, buffer_eval in zip(self.encoder_q.buffers(), self.ema.buffers()):
            buffer_eval.copy_(buffer_train)

    @torch.no_grad()
    def distribution_alignment(self, probs):
        probs_bt_mean = probs.mean(0)
        ptr = int(self.DA_ptr)
        if torch.distributed.get_world_size() > 1:
            torch.distributed.all_reduce(probs_bt_mean)
            self.DA_queue[ptr] = probs_bt_mean / torch.distributed.get_world_size()
        else:
            self.DA_queue[ptr] = probs_bt_mean
        self.DA_ptr[0] = (ptr + 1) % self.DA_len
        probs = probs / self.DA_queue.mean(0)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs.detach()

    def forward(self, im_x, im_u_w=None, im_u_s=None, args=None):
        self.momentum_update_ema()
        bx = im_x.shape[0]
        bu = im_u_w.shape[0]

        logits, logits_u_s_aux = self.encoder_q(torch.cat([im_x, im_u_w, im_u_s]), bs=bx+bu)
        logits_x, logits_u_w, logits_u_s = logits[:bx], logits[bx:bx+bu], logits[bx+bu:]
        
        prob_u_w = F.softmax(logits_u_w, dim=-1)
        if args.DA:
            prob_u_w = self.distribution_alignment(prob_u_w)
        
        return logits_x, prob_u_w, logits_u_s, logits_u_s_aux


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output
