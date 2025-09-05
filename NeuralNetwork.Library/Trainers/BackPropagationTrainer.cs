using NeuralNetwork.Abstration;

namespace NeuralNetwork.Library.Trainers
{
    public static class BackPropagationTrainer
    {
        public static double ApplyBackPropagation(INeuralNetwork network, double[] inputs, double[] target, double learningRate, Func<double[], double[]> predictDelegate)
        {
            // Forward pass
            double[] outputs = predictDelegate(inputs);

            double loss = 0.0;

            // ======== Capa de salida ========
            for (int i = 0; i < network.OutputLayer.Length; i++)
            {
                INeuron neuron = network.OutputLayer[i];
                double o = neuron.OutputValue;
                double y = target[i];

                // Gradiente con BCE simplificado
                double delta = o - y;
                neuron.Delta = delta;

                // Actualizar bias
                neuron.SetBiasValue(neuron.Bias - learningRate * delta);

                // Pérdida con Binary Cross-Entropy
                // Evitamos log(0) con epsilon
                double epsilon = 1e-15;
                loss += -(y * Math.Log(o + epsilon) + (1 - y) * Math.Log(1 - o + epsilon));

                // Actualizar pesos desde última capa oculta
                for (int j = 0; j < network.HiddenLayers.Last().Length; j++)
                {
                    INeuron hiddenNeuron = network.HiddenLayers.Last()[j];
                    double currentWeight = hiddenNeuron.Axon.Terminals[i].Weight;
                    double newWeight = currentWeight - (learningRate * delta * hiddenNeuron.OutputValue);
                    hiddenNeuron.Axon.Terminals[i].SetWeightValue(newWeight);
                }
            }

            // ======== Capas ocultas (backprop) ========
            INeuron[] nextLayer = network.OutputLayer;
            for (int layerIndex = network.HiddenLayers.Length - 1; layerIndex >= 0; layerIndex--)
            {
                for (int neuronIndex = 0; neuronIndex < network.HiddenLayers[layerIndex].Length; neuronIndex++)
                {
                    INeuron currentNeuron = network.HiddenLayers[layerIndex][neuronIndex];

                    // Error acumulado de la siguiente capa
                    double error = 0.0;
                    for (int nextNeuronIndex = 0; nextNeuronIndex < nextLayer.Length; nextNeuronIndex++)
                    {
                        error += currentNeuron.Axon.Terminals[nextNeuronIndex].Weight * nextLayer[nextNeuronIndex].Delta;
                    }

                    // Delta con derivada de activación
                    double delta = error * currentNeuron.Activationfunction().CalculateDerivative(currentNeuron.OutputValue);
                    currentNeuron.Delta = delta;

                    // Actualizar bias
                    currentNeuron.SetBiasValue(currentNeuron.Bias - learningRate * delta);

                    // Pesos desde la capa anterior
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

            // Devolvemos la pérdida total (BCE) para esta muestra
            return loss;
        }
    }
}
