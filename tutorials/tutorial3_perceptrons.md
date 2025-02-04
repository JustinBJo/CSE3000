# Tutorial 3 - Perceptrons

## After this tutorial, you will be able to:

- Understand the pipeline of training artificial neural networks
- Calculate the output of a single perceptron
- Explain the advantages/disadvantages of neural networks

## **Chapter 1: What is a Perceptron?**

A perceptron is like a decision-making machine that mimics how neurons in the brain work.

**Biological Neurons**:

![image.png](Tutorial%203%20-%20Perceptrons%2013e92a48207380878231d22e90410013/70a88038-d690-4995-abf2-e4ec8d444e61.png)

*Source: [https://appliedgo.net/perceptron/](https://appliedgo.net/perceptron/)*

In your brain, a neuron receives information from other neurons, processes it, and decides whether to pass the information along. For example, if you touch something hot, your neurons process this information and decide to make you pull your hand away.

**Artificial Neurons**:

![image.png](Tutorial%203%20-%20Perceptrons%2013e92a48207380878231d22e90410013/87e309eb-8798-4cce-b3fa-5efce47ea78f.png)

*Source: [https://appliedgo.net/perceptron/](https://appliedgo.net/perceptron/)*

Similarly, a perceptron receives multiple input values, processes them mathematically, and generates an output. This process involves:

1. **Input Information**: A perceptron takes in information, like numbers or data (e.g., temperature or cloud cover).
2. **Weighting**: It gives more importance to some information than others, like how you might care more about temperature than cloud cover when deciding if it’s sunny.
3. **Summation**: It adds up all the information after giving it the proper importance (weights).
4. **Decision**: It checks whether the sum is big enough to “fire” a signal. If it is, it outputs one answer; if not, it outputs another.

**Example**:

Imagine a perceptron designed to predict whether it will rain based on three inputs: temperature, humidity and cloud cover. Each input is assigned a weight based on its importance, and the perceptron computes whether it will rain or not.

![image.png](Tutorial%203%20-%20Perceptrons%2013e92a48207380878231d22e90410013/image.png)

## 2. Chapter 2: How does a perceptron learn?

### 2.1. Structure of a Perceptron

![afbeelding.png](Tutorial%203%20-%20Perceptrons%2013e92a48207380878231d22e90410013/afbeelding.png)

A perceptron consists of:

- **Inputs**
    - A vector representing features of the data
    
    $$
    x=(x_{1}, x_{2},...,x_{l})
    $$
    

- **Weights**
    - Multiplied element-wise by input values
    - Determined by the importance of a feature
        - For example, if cloud cover is a more important feature to predict rain than temperature, the weight for cloud cover will be greater than that for temperature.

$$
w=(w_{1}, w_{2},...,w_{l})
$$

- **Bias**
    - An additional parameter that helps adjust the value so the perceptron makes better decisions
- **Summation**
    - Adds up all inputs multiplied by their weights and the bias
    
    $$
    Sum=w_{1}⋅x_{1}+w_{2}⋅x_{2}+...+w_{l}⋅x_{l}+b
    $$
    
- **Activation Function**
    - Takes the summation value and determines the output of the perceptron
    - There are many different types of activation functions
        - example: step function

$$
f(x) =\begin{cases}1&\text{if $x\geq0$}\\0&\text{if $x<0$}\end{cases}
$$

### 2.2. Training a Perceptron

Training involves adjusting the weights and bias to minimize errors in predicting the output. This is done using a method called **gradient descent**:

1. **Initialization**: Start with random weights and bias.
2. **Prediction**: For each training example, calculate the perceptron's output.
3. **Update Weights**: Adjust the weights and bias based on the error (difference between predicted and actual outputs).
    
    $$
    w_i←w_i+η⋅(y−\hat{y})⋅x_i
    $$
    
    - $*w_i$* : Current weight for input xi.
    - $*\eta$* : Learning rate (a small positive number that decides how rapidly the weight is going to change in training).
    - $*y$* : Actual output.
    - $*\hat{y}$* : Predicted output.
4. **Repeat**: Continue this process for all training data until the error is minimized.

**Example**:

Training a perceptron to classify whether a flower is a rose based on petal length (x1) and petal width (x2). If the model predicts incorrectly, the weights for x1 and x2 are adjusted slightly to improve predictions.

## Chapter 3: Multi-Layer Perceptron

### 3.1. Structure of the Multi-Layer Perceptron

As seen in the first example, a single perceptron could be used to predict very simplified problems. We can however use this simple structure to learn very complex tasks by combining them in multiple layers:

![1_4_BDTvgB6WoYVXyxO8lDGA.png](Tutorial%203%20-%20Perceptrons%2013e92a48207380878231d22e90410013/1_4_BDTvgB6WoYVXyxO8lDGA.png)

This is the Multi-Layer Perceptron (MLP), and is the most basic form of an Artificial Neural Network (ANN). The blue nodes and yellow nodes are all single perceptrons, only their inputs and outputs are linked to other perceptrons. For MLPs, all nodes in a layer are connected to all other nodes in the next layer. This will allow each subsequent layer to find more complex patterns.

The pink input layer is exactly the same as in the single perceptron part, only they are connected to **all** nodes in the first layer.

The output layer also consist of perceptrons, and can have any number of nodes, depending on what the model wants to achieve. For instance, if we have a classification problem, where the goal is to decide what a picture represents, we might have three output nodes: cat, dog and bird, each representing the likelihood of the picture being the animal. We may also just have a single output node, like with the original example of “is it going to rain”.

**Example**:

Let’s say we want a model to be able to recognise handwritten digits. This can be represented as an input of 256 features (for a picture of 16x16 pixels). For us it is a trivial task to tell which digit an input is, but this is because we can easily see patterns in the writing, like a circle with a line extending beneath it for 9. However, as you can see below, the data for two distinct 9s may have very little white pixels shared between them:

![MNIST dataset](Tutorial%203%20-%20Perceptrons%2013e92a48207380878231d22e90410013/MNIST_dataset_example.png)

MNIST dataset

We will use a MLP with two hidden layers, each containing 16 nodes, and an output layer with 10 nodes, each representing the likelihood of the input being one of the digits. 

After training with the gradient descent method, the nodes in the first hidden layer will use some combination of pixels in certain areas, combining to get **new** features (many pixels in the middle of the picture, or little pixels in the bottom left). These features are then sent to those in the second layer to give **even more complex** features (loops in bottom half, or vertical line through the center). The introduction of multiple layers allow the network to learn complex patterns in the data!

### 3.2 Advantages/disadvantages of Neural Networks

Artificial Neural Network have some big advantages as opposed to other learning methods:

- Suited to complex problems
    - ANNs can find patterns that are hard to find by other learning models
- Versatile in their use: apart from some examples in this tutorial, neural networks are used in a huge variety of tasks:
    - Image Processing: the structure of the networks is well suited to images (arrays of pixels) or even video.
    - Linguistics: the same reasoning holds for text-based inputs. This is why Large Language Models like ChatGPT have made large strides over the past years.

But also some downsides:

- Needs much computing power, for many rounds of training
- Large amounts of training data necessary
- Acts as a “black box”
    - Large networks can be seen as black boxes, meaning that we only really see the input and output. The hidden layers find complex patterns from the input data, but especially in networks with many layers, it becomes very difficult to see what those complex new features are, and what caused the system to give any specific output.
