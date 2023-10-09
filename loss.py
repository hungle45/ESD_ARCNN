import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin, mode='euclid'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
        if mode == 'euclid':
            self.sim = self.euclid_sim
        elif mode == 'cosine':
            self.sim = self.cosine_sim
        else:
            raise Exception('`mode` has invalid value')

    def euclid_sim(self, output1, output2, target):
        eps = 1e-9
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + eps).sqrt()).pow(2))
        return losses.mean()
    
    def cosine_sim(self, output1, output2, target):
        # cosine similarity
        cos_func = torch.nn.CosineSimilarity(-1)
        cosine = cos_func(output1, output2)

        loss_similarity = 1 - cosine
        loss_dissimlarity = torch.clamp(cosine - self.margin, min=0.0)

        loss = target * loss_similarity + (1 - target) * loss_dissimlarity
        loss = torch.sum(loss) / output1.size()[0]

        return loss
    
    def forward(self, output1, output2, target):
        return self.sim(output1, output2, target)