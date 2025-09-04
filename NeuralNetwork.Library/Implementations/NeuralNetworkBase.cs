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
            //TODO
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
