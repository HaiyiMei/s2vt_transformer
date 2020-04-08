import math
from torch.utils.data.sampler import WeightedRandomSampler

class ScheduleSample():
    def __init__(self, mode, thresh, k, c):
        """
        :param mode:0:Linear_decay
                    1:Exponential_decay
                    2:Inverse_sigmoid_decay
        :param thresh: minimum prob of gt caption
        :param k: a constant to modify the speed of decay
        :param c: c
        """
        self.mode = mode
        self.thresh = thresh
        self.k = k
        self.c = c

    def get_prob(self, epoch):
        """
        :param epoch: current epoch
        :return: sample_prob
        """
        if self.mode == 0:
            return max(self.thresh, self.k-self.c*epoch)
        elif self.mode == 1:
            return self.k**(epoch/100)
        elif self.mode == 2:
            return self.k/(self.k + math.exp(epoch/self.k))
        else:
            print('mode error')

    def schedule_sample(self, epoch):
        """

        :param epoch:current epoch
        :return: index to chose gt or generate(0 is gt and 1 is generate)
        """
        sample_prob =self.get_prob(epoch)
        index = list(WeightedRandomSampler([sample_prob, 1-sample_prob], 1, replacement=True))[0]
        return index