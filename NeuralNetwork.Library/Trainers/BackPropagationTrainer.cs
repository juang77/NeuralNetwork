using NeuralNetwork.Abstration;

namespace NeuralNetwork.Library.Trainers
{
    public static class BackPropagationTrainer
    {
        public static double ApplyBackPropagation(INeuralNetwork network, double[] inputs, double[] target, double learningRate)
        {
            double loss = 0.0;

            // ======== Output Layer ========
            for (int i = 0; i < network.OutputLayer.Length; i++)
            {
                INeuron neuron = network.OutputLayer[i];
                double o = neuron.OutputValue;
                double y = target[i];

                // Simplified gradient with BCE
                double delta = o - y;
                neuron.Delta = delta;

                // Update bias
                neuron.SetBiasValue(neuron.Bias - learningRate * delta);

                // Loss with Binary Cross-Entropy
                // Avoid log(0) using epsilon
                double epsilon = 1e-15;
                loss += -(y * Math.Log(o + epsilon) + (1 - y) * Math.Log(1 - o + epsilon));

                // Update weights from last hidden layer
                for (int j = 0; j < network.HiddenLayers.Last().Length; j++)
                {
                    INeuron hiddenNeuron = network.HiddenLayers.Last()[j];
                    double currentWeight = hiddenNeuron.Axon.Terminals[i].Weight;
                    double newWeight = currentWeight - (learningRate * delta * hiddenNeuron.OutputValue);
                    hiddenNeuron.Axon.Terminals[i].SetWeightValue(newWeight);
                }
            }

            // ======== Hidden layers (backprop) ========
            INeuron[] nextLayer = network.OutputLayer;
            for (int layerIndex = network.HiddenLayers.Length - 1; layerIndex >= 0; layerIndex--)
            {
                for (int neuronIndex = 0; neuronIndex < network.HiddenLayers[layerIndex].Length; neuronIndex++)
                {
                    INeuron currentNeuron = network.HiddenLayers[layerIndex][neuronIndex];

                    // Accumulated error from the next layer
                    double error = 0.0;
                    for (int nextNeuronIndex = 0; nextNeuronIndex < nextLayer.Length; nextNeuronIndex++)
                    {
                        error += currentNeuron.Axon.Terminals[nextNeuronIndex].Weight * nextLayer[nextNeuronIndex].Delta;
                    }

                    // Delta with activation derivative
                    double delta = error * currentNeuron.Activationfunction().CalculateDerivative(currentNeuron.OutputValue);
                    currentNeuron.Delta = delta;

                    // Update bias
                    currentNeuron.SetBiasValue(currentNeuron.Bias - learningRate * delta);

                    // Weights from the previous layer
                    INeuron[] previousLayer = layerIndex == 0 ? network.InputLayer : network.HiddenLayers[layerIndex - 1];

                    for (int prevNeuronIndex = 0; prevNeuronIndex < previousLayer.Length; prevNeuronIndex++)
                    {
                        INeuron previousNeuron = previousLayer[prevNeuronIndex];
                        double currentWeight = previousNeuron.Axon.Terminals[neuronIndex].Weight;
                        double newWeight = currentWeight - (learningRate * delta * previousNeuron.OutputValue);
                        previousNeuron.Axon.Terminals[neuronIndex].SetWeightValue(newWeight);
                    }
                }

                nextLayer = network.HiddenLayers[layerIndex];
            }

            // Return total loss (BCE) for this sample
            return loss;
        }
    }
}
