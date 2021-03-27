# Transfer-learning-caltech


I started to explore transfer learning, the concept of using a pre-trained Convo- lutional Neural Network to classify class labels outside of what it was originally trained on. In general, there are two methods to perform transfer learning when applied to deep learning and computer vision:
1. Treat networks as feature extractors, forward propagating the image until a given layer, and then taking these activations and treating them as feature vectors.

 2. Fine-tuning networks by adding a brand-new set of fully-connected layers to the head of the network and tuning these FC layers to recognize new classes (while still using the same underlying CONV filters).
I focused strictly on the feature extraction component of transfer learning in this chapter, demonstrating that deep CNNs such as VGG, Inception, and ResNet are capable of acting as powerful feature extraction machines, even more powerful than hand-designed algorithms such as HOG ,SIFT, and Local Binary Patterns just to name a few. Whenever approaching a new problem with deep learning and Convolutional Neural Networks, always consider if applying feature extraction will obtain reasonable accuracy – if so,we can skip the network training process entirely, saving us a ton of time, effort, and headache.
