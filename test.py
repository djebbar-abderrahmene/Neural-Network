import pickle
import numpy as np

# Laoding Training Data
def load_training_data():
    with open('data/labels.pkl','rb') as f:
        labels = pickle.load(f)

    with open(f"data/images.pkl",'rb') as f:
        images = pickle.load(f)

    return list(zip(images,labels))


class Neural_Network :
    def __init__(self):
        # The Neural Network Layers
        self.layer = [[],[],[],[]]
        self.layer[0] = np.zeros(784)
        self.layer[1] = np.zeros(16)
        self.layer[2] = np.zeros(16)
        self.layer[3] = np.zeros(10)


        

        # The Weight Matricies 
        
        self.W = [[],[],[]]
        np.random.seed(0)  # Set seed for reproducibility

        # Xavier/Glorot Initialization
        self.W[0] = np.random.randn(16, 784) * np.sqrt(2/(784 + 16))  # Consider both input and output dimensions
        self.W[1] = np.random.randn(16, 16) * np.sqrt(2/(16 + 16))
        self.W[2] = np.random.randn(10, 16) * np.sqrt(2/(16 + 10))


        # Biases Vectors
        self.B = [[],[],[]]
        self.B[0] = np.zeros(16)
        self.B[1] = np.zeros(16)
        self.B[2] = np.zeros(10)



    def _sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def _sigmoidP(self,x):
        return np.exp(-x)/((1+np.exp(-x))**2)

    # The output layer that was supposed to be true
    def y(self,label):
        a = np.zeros_like(self.layer[3])
        a[label] = 1
        return a


    def test(self,test_data) :
        c = 0
        for i in range(len(test_data)):
            self.feedforward(test_data[i][0])

            if (np.argmax(self.layer[3]) == test_data[i][1]):
                c += 1

        acc = (c / len(test_data)) * 100

        return acc



    #Uses Stochastic Gradient Descent
    def train(self,training_data,epochs,eta):
        mb_size = 125
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1} Starts \n")

            # Dividing The Training data into mini batches
            np.random.shuffle(training_data)
            mini_batches = [ training_data[k:k+mb_size]  for k in range(len(training_data)//mb_size)]
            for mini_batch in mini_batches:   
                self.gradient_step(mini_batch,eta)

    # caculate the gradient according to the mini batch, and update the weights and biases.
    def gradient_step(self,mini_batch,eta):
        # Gradient Vector (For Mini Batch) Initilizing
        self.gw = [np.zeros_like(self.W[i]) for i in range(3)]
        self.gb = [np.zeros_like(self.B[i]) for i in range(3)]

        for image, label in mini_batch :
            # Gradient Vector for a single Training Example Intilization 
            gw0 = [np.zeros_like(self.W[i]) for i in range(3)]
            gb0 = [np.zeros_like(self.B[i]) for i in range(3)]

            # Set the layer outputs
            self.feedforward(image)

            
            # Gradient of the activations, we will use it to backpropagate
            ga = 2*(self.layer[3]-self.y(label)) 

            # max index for weights and biases.

            i = 2
            # Note That since the activations and the parameters don't have the same index
            # so a layer with index i, coresponds to bias with index i-1
            while i>=0 :
                # Caculating Z
                z = self.W[i]@self.layer[i]+self.B[i]


                gw0[i] = np.outer(self._sigmoidP(z)*ga,self.layer[i])

                gb0[i] = self._sigmoidP(z)*ga

                # using ga to backpropagate
                ga = (self.W[i].T)@(self._sigmoidP(z)*ga)

                i = i-1


            # Accumilating all of the gradients    
            self.gw = [self.gw[i] + gw0[i] for i in range(3)]
            self.gb = [self.gb[i] + gb0[i] for i in range(3)]        




        # getting the average by diving by the number of training example

        self.gw = [self.gw[i] / len(mini_batch)  for i in range(3)]
        self.gb = [self.gb[i] / len(mini_batch)  for i in range(3)]

        self.W = [self.W[i] - eta * self.gw[i] for i in range(3)]
        self.B = [self.B[i] - eta * self.gb[i] for i in range(3)]


    def feedforward(self,input):
        self.layer[0] = input
        # Using the neural Network to determine the wirtten degit
        for i in range(3):
            z = self.W[i] @ self.layer[i]
            z = z + self.B[i]
            self.layer[i+1] = self._sigmoid(z)




NN = Neural_Network()


NN.train(load_training_data(),100,0.005)

