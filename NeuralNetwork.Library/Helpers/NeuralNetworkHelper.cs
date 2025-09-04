using NeuralNetwork.Abstration;
using NeuralNetwork.Library.Implementations;

namespace NeuralNetwork.Library.Helpers
{
    public static class NeuralNetworkHelper
    {
        public static IEnumerable<NeuronInfo> GetNeuronInfo(this INeuralNetwork neuralNetwork)
        {
            if (neuralNetwork == null)
                throw new ArgumentNullException(nameof(neuralNetwork));

            List<NeuronInfo> result = [];

            int currentLayerIndex = 0;
            
            //Capa de entrada
            AddInfo(neuralNetwork.InputLayer, currentLayerIndex++);
            
            //Capas ocultas
            foreach (var hiddenLayer in neuralNetwork.HiddenLayers)
            {
                AddInfo(hiddenLayer, currentLayerIndex++);
            }

            //Capa de salida
            AddInfo(neuralNetwork.OutputLayer, currentLayerIndex);

            return result;


            void AddInfo(INeuron[] layer, int layerIndex)
            {
                for (int i = 0; i < layer.Length; i++)
                {
                    result.Add(new NeuronInfo(layerIndex, i, layer[i].Axon.Terminals.Select(t => t.Weight).ToArray(), layer[i].Bias, layer[i].OutputValue));
                }
            }
            
        }

        public static INeuron[] CreateInputLayer(int neuronsCount, IInputFunction inputFunction, IActivationFunction activationFunction)
        {
            if (neuronsCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(neuronsCount), "The number of neurons must be greater than zero.");

            INeuron[] layer = new INeuron[neuronsCount];

            Synapse Dendrite;
            Neuron Neuron;

            for (int i = 0; i < neuronsCount; i++)
            {
                Dendrite = new Synapse();
                Neuron = new Neuron(inputFunction, activationFunction);
                Neuron.AddDendrite(Dendrite);
                layer[i] = Neuron;
            }

            return layer;
        }


        private static INeuron[] CreateLayer(INeuron[] previousLayer, int neuronsCount, IInputFunction inputFunction, IActivationFunction activationFunction)
        {
            if (neuronsCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(neuronsCount), "The number of neurons must be greater than zero.");
            if (previousLayer == null || previousLayer.Length == 0)
                throw new ArgumentNullException(nameof(previousLayer), "The previous layer must not be null or empty.");

            INeuron[] NewLayer = new INeuron[neuronsCount];

            for (int i = 0; i < neuronsCount; i++)
            {
                NewLayer[i] = new Neuron(inputFunction, activationFunction);

            }

            Synapse Synapse;

            // Connect to previous layer
            foreach (INeuron prevNeuron in previousLayer)
            {
               foreach (var postNeuron in NewLayer)
               {
                    Synapse = new Synapse();
                    prevNeuron.AddTerminal(Synapse);
                    postNeuron.AddDendrite(Synapse);
               }
            }

            
            return NewLayer;
        }

        public static INeuron[][] CreateHiddenLayer(int[] hiddenLayersNeuronsCount, INeuron[] inputLayer, IInputFunction inputFunction, IActivationFunction activationFunction)
        {
            if (hiddenLayersNeuronsCount == null || hiddenLayersNeuronsCount.Length == 0)
                throw new ArgumentNullException(nameof(hiddenLayersNeuronsCount), "Hidden layers configuration must not be null or empty.");
            if (inputLayer == null || inputLayer.Length == 0)
                throw new ArgumentNullException(nameof(inputLayer), "Input layer must not be null or empty.");

            INeuron[][] hiddenLayers = new INeuron[hiddenLayersNeuronsCount.Length][];
            INeuron[] previousLayer = inputLayer;

            for (int i = 0; i < hiddenLayersNeuronsCount.Length; i++)
            {
                int neuronsCount = hiddenLayersNeuronsCount[i];
                hiddenLayers[i] = CreateLayer(previousLayer, neuronsCount, inputFunction, activationFunction);
                previousLayer = hiddenLayers[i];
            }

            return hiddenLayers;
        }
        public static INeuron[] CreateOutputLayer(int neuronsCount, INeuron[] previousLayer, IInputFunction inputFunction, IActivationFunction activationFunction)
        {
            if (neuronsCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(neuronsCount), "The number of neurons must be greater than zero.");
            if (previousLayer == null || previousLayer.Length == 0)
                throw new ArgumentNullException(nameof(previousLayer), "The previous layer must not be null or empty.");


            INeuron[] outputLayer = new INeuron[neuronsCount];

            Synapse Synapse;

            for (int i = 0; i < neuronsCount; i++)
            {
                var neuron = new Neuron(inputFunction, activationFunction);

                foreach (var prevNeuron in previousLayer)
                {
                    Synapse = new Synapse();
                    prevNeuron.AddTerminal(Synapse);
                    neuron.AddDendrite(Synapse);
                }

                outputLayer[i] = neuron;
            }

            return outputLayer;
        }
        
    }
}
