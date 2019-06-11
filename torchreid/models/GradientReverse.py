from torch.autograd import Function

class GradReverse(Function):
    @staticmethod
    def forward(self, x, lambd):
        self.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        #return (grad_output * -self.lambd)
        return grad_output.neg()*self.lambd, None


def grad_reverse(x, lambd):
    GradReverse.lambd= lambd
    return GradReverse().apply(x, lambd)
