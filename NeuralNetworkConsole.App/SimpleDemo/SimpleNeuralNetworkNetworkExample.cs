using NeuralNetwork.Abstration;

namespace NeuralNetworkConsole.App.SimpleDemo
{
    internal static class SimpleNeuralNetworkNetworkExample
    {
        public static void DoExample()
        {
            INeuralNetwork neuralNetwork = new SimpleNeuralNetwork();
            var output = neuralNetwork.Predict([1,2]);
            WriteNeuronsInfo(neuralNetwork);
            var probabilities = SoftMax(output);
            foreach (var p in probabilities)
            {
                Console.WriteLine($"Probability: {p:f4}");
            }
        }

        public static double[] SoftMax(double[] Outputs)
        {
            var expValues = Outputs.Select(v => Math.Exp(v));
            var sumExpValues = expValues.Sum();
            var probabilities = expValues.Select(v => v / sumExpValues);
            return probabilities.ToArray();
        }


        public static void WriteNeuronsInfo(INeuralNetwork neuralNetwork)
        { 
            var NeuronsInfo = neuralNetwork.GetNeuronInfo();
            int currentLayer = -1;
            int CurrentNeuronIndex = -1;

            foreach (var neuronInfo in NeuronsInfo)
            {
                if(CurrentNeuronIndex != neuronInfo.LayerIndex)
                {
                    currentLayer = neuronInfo.LayerIndex;
                    Console.WriteLine($"Layer: {currentLayer}");
                    CurrentNeuronIndex = -1;
                }
                
                CurrentNeuronIndex = neuronInfo.NeuronIndex;
                Console.WriteLine($"  Neuron Index {CurrentNeuronIndex}");
                Console.WriteLine($"    Output Value: {neuronInfo.OutputValue:f4}");
                Console.WriteLine($"    Bias: {neuronInfo.Bias:f4}");

                if(neuronInfo.Weights != null && neuronInfo.Weights.Length > 0)
                {
                    Console.WriteLine($"    Weights: ");
                    for (int i = 0; i < neuronInfo.Weights.Length; i++)
                    { 
                        Console.WriteLine($"      Synapse {i}: {neuronInfo.Weights.ElementAt(i):f2}");
                    }
                }
            }
        }
    }
}
