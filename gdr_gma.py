import torch
import torch.nn as nn

class MemoryBank:
    def __init__(self, size):
        self.grads = []    
        self.size = size
        
    def update(self, grads):
        self.grads.append(grads)
        if len(self.grads) > self.size:
            del self.grads[0]

    def get_graident(self, model:nn.Module):
        gradient = []
        for _, param in model.named_parameters():
            if param.requires_grad:
                grad = param.grad.clone().detach()
                gradient.append(grad.view(-1))

        return gradient  
    
    def mean_grads(self, t_grads):
        grads = []
        for grad in self.grads:
            if torch.cosine_similarity(grad, t_grads, dim=0) < 0:
                grads.append(grad)
        if len(grads) > 0:
            avg_grad = grads[0]
            for grad in grads[1:]:
                avg_grad += grad
            avg_grad = avg_grad/len(grads) 

            return avg_grad
    
        else:
            return None

def get_gradient(model:nn.Module):
    gradient = []
    for _, param in model.named_parameters():
        if param.requires_grad:
            grad = param.grad.clone().detach()
            gradient.append(grad.view(-1))

    return gradient    


def rectify_graident(grads_x, grads_y):
    r_grads_x = []
    r_grads_y = []

    for x, y in zip(grads_x, grads_y):
        if torch.cosine_similarity(x, y, dim=0) < 0:
            InP_xy = torch.matmul(y, x) 
            Inp_xx = torch.norm(x, p=2) ** 2
            Inp_yy = torch.norm(y, p=2) ** 2
            x = x - InP_xy/Inp_yy * y
            y = y - InP_xy/Inp_xx * x

        r_grads_x.append(x)
        r_grads_y.append(y)

    return r_grads_x, r_grads_y
    