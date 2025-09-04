using System.Data;

namespace NeuralNetwork.Abstration
{
    public interface INeuralNetwork
    {
        INeuron[] InputLayer { get; }

        INeuron[][] HiddenLayers { get; }

        INeuron[] OutputLayer { get; }

        double[] Predict (double[] input);

        void Train(double[][] trainingData, double[][] targets, int epochs, double learningRate);

        IEnumerable<NeuronInfo> GetNeuronInfo();
        
        void SaveModel(string filePath);

        INeuralNetwork LoadModel(string filePath);
    }
}
