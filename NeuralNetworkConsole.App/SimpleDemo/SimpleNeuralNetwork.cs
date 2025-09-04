using NeuralNetwork.Abstration;
using NeuralNetwork.Library.Implementations;

namespace NeuralNetworkConsole.App.SimpleDemo
{
    public class SimpleNeuralNetwork : NeuralNetworkBase
    {
        public SimpleNeuralNetwork()
        {
            CreateNeuralNetwork(2, [4,2,2], 3);
        }

        public override double[] Predict(double[] input)
        {
            int InputIndex = 0;
            foreach (var dendrite in InputDendrites)
            { 
                dendrite.ReceiveInputValue(input[InputIndex++]);
            }
            var Outputs = OutputLayer.Select(n => n.OutputValue).ToArray();
            return Outputs;
        }

        public override void Train(double[][] trainingData, double[][] targets, int epochs, double learningRate)
        {
            throw new NotImplementedException();
        }

        protected override void Initialize()
        {
            int Value = 1;
            // Initialize weights for all synapses from input layer to first hidden layer
            foreach (INeuron neuron in InputLayer)
            {
                foreach (ISynapse Synapse in neuron.Axon.Terminals)
                {
                    Synapse.SetWeightValue(Value++ /100.0);
                }
            }
            // Initialize weights for all synapses from hidden layers to next hidden layer or output layer
            foreach (INeuron[] layer in HiddenLayers)
            {
                foreach (INeuron neuron in layer)
                {
                    foreach (ISynapse Synapse in neuron.Axon.Terminals)
                    {
                        Synapse.SetWeightValue(Value++ / 100.0);
                    }
                }
            }
            //Initialize Biases for all neurons in hidden layers and output layer
            foreach (INeuron[] layer in HiddenLayers)
            {
                foreach (INeuron neuron in layer)
                {
                    neuron.SetBiasValue(Value++ / 100.0);
                }
            }
            // Initialize Biases for all neurons in output layer
            foreach (INeuron neuron in OutputLayer)
            {
                neuron.SetBiasValue(Value++ / 100.0);
            }
        }
    }
}
