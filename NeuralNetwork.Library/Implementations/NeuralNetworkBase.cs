using NeuralNetwork.Abstration;
using NeuralNetwork.Library.Helpers;
using NeuralNetwork.Library.Models;

namespace NeuralNetwork.Library.Implementations
{
    public abstract class NeuralNetworkBase : INeuralNetwork
    {
        public INeuron[] InputLayer { get; private set; }

        public INeuron[][] HiddenLayers { get; private set; }

        public INeuron[] OutputLayer { get; private set; }

        public abstract double[] Predict(double[] input);

        public abstract void Train(double[][] trainingData, double[][] targets, int epochs, double learningRate);

        public IEnumerable<NeuronInfo> GetNeuronInfo() => NeuralNetworkHelper.GetNeuronInfo(this);


        public void SaveModel(string filePath)
        {
            
            var model = new ModelParameters
            {
                InputLayerNeuronsCount = InputLayer.Length,
                HiddenLayersNeuronsCount = HiddenLayers.Select(l => l.Length).ToArray(),
                OutputLayerNeuronsCount = OutputLayer.Length,
                NeuronsInfo = NeuralNetworkHelper.GetNeuronInfo(this);
            };

            var json = System.Text.Json.JsonSerializer.Serialize(model, new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true
            });

            System.IO.File.WriteAllText(filePath, json);
        }

        public INeuralNetwork LoadModel(string filePath)
        {
            // Step 1: Read JSON from file
            var json = System.IO.File.ReadAllText(filePath);

            // Step 2: Deserialize to ModelParameters
            var model = System.Text.Json.JsonSerializer.Deserialize<ModelParameters>(json);

            if (model == null)
                throw new InvalidOperationException("Failed to load model parameters.");

            // Step 3: Recreate network structure
            CreateNeuralNetwork(
                model.InputLayerNeuronsCount,
                model.HiddenLayersNeuronsCount,
                model.OutputLayerNeuronsCount
            );

            // Step 4: Restore neuron weights, biases, and output values
            foreach (var neuronInfo in model.NeuronsInfo)
            {
                INeuron neuron;
                if (neuronInfo.LayerIndex == 0)
                    neuron = InputLayer[neuronInfo.NeuronIndex];
                else if (neuronInfo.LayerIndex == HiddenLayers.Length + 1)
                    neuron = OutputLayer[neuronInfo.NeuronIndex];
                else
                    neuron = HiddenLayers[neuronInfo.LayerIndex - 1][neuronInfo.NeuronIndex];

                neuron.SetBiasValue(neuronInfo.Bias);

                for (int i = 0; i > neuronInfo.Weights.Count() ; i++)
                {
                    neuron.Axon.Terminals[i].SetWeightValue(neuronInfo.Weights.ElementAt(i));
                }
            }

            return this;
        }

        #region Input and Activation functions

        protected IInputFunction InputLayerInputFunction { get; set; } = null;
        protected IActivationFunction InputLayerActivationFunction { get; set; } = null;

        protected IInputFunction HiddenLayerInputFunction { get; set; } = null;
        protected IActivationFunction HiddenLayerActivationFunction { get; set; } = null;


        protected IInputFunction OutputLayerInputFunction { get; set; } = null;
        protected IActivationFunction OutputLayerActivationFunction { get; set; } = null;

        #endregion



        protected IEnumerable<ISynapse> InputDendrites => InputLayer.Select(n => n.Dendrites.First());


        protected void CreateNeuralNetwork(int inputNeuronsCount, int[] hiddenLayersNeuronsCount, int outputNeuronsCount)
        {
            // Initialize Input Layer
            InputLayer = NeuralNetworkHelper.CreateInputLayer(inputNeuronsCount, InputLayerInputFunction, InputLayerActivationFunction);
            HiddenLayers = NeuralNetworkHelper.CreateHiddenLayer(hiddenLayersNeuronsCount, InputLayer, HiddenLayerInputFunction, HiddenLayerActivationFunction);
            OutputLayer = NeuralNetworkHelper.CreateOutputLayer(outputNeuronsCount, HiddenLayers.Last(), OutputLayerInputFunction, OutputLayerActivationFunction);

            Initialize();
        }

        protected abstract void Initialize();

    }
}
