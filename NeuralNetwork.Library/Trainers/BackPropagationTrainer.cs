using NeuralNetwork.Abstration;

namespace NeuralNetwork.Library.Trainers
{
    public static class BackPropagationTrainer
    {
        public static double ApplyBackPropagation(INeuralNetwork network, double[] inputs, double[] target, double learningRate, Func<double[], double[]> predictDelegate)
        {
            // Forward pass
            double[] outputs = predictDelegate(inputs);
            // Calculate loss (Mean Squared Error)
            double loss = 0.0;
            for (int i = 0; i < network.OutputLayer.Length; i++)
            {
                INeuron neuron = network.OutputLayer[i];
                double Error = neuron.OutputValue - target[i];
                double Delta = Error * neuron.Activationfunction().CalculateDerivative(neuron.OutputValue);
                neuron.Delta = Delta;

                neuron.SetBiasValue(neuron.Bias - learningRate * Delta);

                loss += Error * Error;

                for (int j = 0; j < network.HiddenLayers.Last().Length; j++)
                {
                    INeuron hiddenNeuron = network.HiddenLayers.Last()[j];
                    double currentWeight = hiddenNeuron.Axon.Terminals[i].Weight;
                    double weightGradient = currentWeight - learningRate * Delta * hiddenNeuron.OutputValue;
                    hiddenNeuron.Axon.Terminals[i].SetWeightValue(weightGradient);
                }
            }

            INeuron[] NextLayer = network.OutputLayer;
            for (int layerIndex = network.HiddenLayers.Length - 1; layerIndex >= 0; layerIndex--)
            {
                for (int NeuronIndex = 0; NeuronIndex < network.HiddenLayers[layerIndex].Length; NeuronIndex++)
                {
                    INeuron CurrentNeuron = network.HiddenLayers[layerIndex][NeuronIndex];
                    double Error = 0.0;
                    for (int NextNeuronIndex = 0; NextNeuronIndex < NextLayer.Length; NextNeuronIndex++)
                    {
                        Error += CurrentNeuron.Axon.Terminals[NextNeuronIndex].Weight * NextLayer[NextNeuronIndex].Delta;
                    }

                    double Delta = Error * CurrentNeuron.Activationfunction().CalculateDerivative(CurrentNeuron.OutputValue);
                    CurrentNeuron.Delta = Delta;
                    CurrentNeuron.SetBiasValue(CurrentNeuron.Bias - learningRate * Delta);

                    INeuron[] PreviousLayer = layerIndex == 0 ? network.InputLayer : network.HiddenLayers[layerIndex - 1];

                    for (int PrevNeuronIndex = 0; PrevNeuronIndex < PreviousLayer.Length; PrevNeuronIndex++)
                    {
                        INeuron previousNeuron = PreviousLayer[PrevNeuronIndex];
                        double currentWeight = previousNeuron.Axon.Terminals[NeuronIndex].Weight;
                        double weightGradient = currentWeight - learningRate * Delta * previousNeuron.OutputValue;
                        previousNeuron.Axon.Terminals[NeuronIndex].SetWeightValue(weightGradient);
                    }
                }
                NextLayer = network.HiddenLayers[layerIndex];
            }
            return loss / network.OutputLayer.Length;
        }
    }
}
