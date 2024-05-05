import numpy as np
class BAM:
    def __init__(self,input_size,output_size):
        self.weights = np.zeros((input_size,output_size))

    def train(self,input_pattern,output_pattern):
        for i in range(input_pattern.shape[0]):
            x = input_pattern[i]
            y = output_pattern[i]
            self.weights += np.outer(x,y)

    def recall_input(self,output_pattern):
        return np.dot(output_pattern , self.weights)

    def recall_output(self,input_pattern):
        return np.dot(input_pattern , self.weights)

input_size = 2   #or simply input_size =2
output_size = 2 #or simply output_size =2
bam = BAM(input_size,output_size)

input_pattern = np.array([[1,-1],[-1,1]])
output_pattern = np.array([[-1,1],[1,-1]])
bam.train(input_pattern , output_pattern)

input_recall = bam.recall_input(output_pattern)
print("Input Recalled:",input_recall )
print("From Output :",output_pattern)

output_recalled = bam.recall_output(input_pattern)
print("Input:",input_pattern)
print("Output Recalled:",output_recalled)
