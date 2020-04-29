from torch import nn

class MergeNet(nn.Module):

    def __init__(self,number_of_trained_people):
        '''

        :param number_of_trained_people: Indicating how many people in training set(which will be the input dimension of MergeNet)
        '''
        super(MergeNet, self).__init__()
        self.input_size = number_of_trained_people
        self.output_size = 1

        self.fc = nn.Linear(self.input_size, self.output_size)

    def forward(self,x_target):
        '''
        Given target points x_target,
        returns predicted value as estimated age.
        (We just combine the result of multiple neural process networks to realize multi-task training)
        '''
        return self.fc(x_target)

