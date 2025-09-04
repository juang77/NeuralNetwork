using NeuralNetwork.Abstration;
using NeuralNetwork.Library.Activation_Function;
using NeuralNetwork.Library.Implementations;
using NeuralNetwork.Library.Initializers;
using NeuralNetwork.Library.Trainers;

namespace NeuralNetworkConsole.App.StudyCase;

public class StudyCaseNeuralNetwork : NeuralNetworkBase, INeuralNetwork
{
    public StudyCaseNeuralNetwork()
    {
        HiddenLayerActivationFunction = new SigmoidActivationfunction();
        OutputLayerActivationFunction = new SigmoidActivationfunction();
        CreateNeuralNetwork(2, [8], 1);
    }

    public override double[] Predict(double[] inputs)
    {
        int inputIndex = 0;
        foreach (var Dendrite in InputDendrites)
        {
            Dendrite.ReceiveInputValue(inputs[inputIndex++]);
        }
        var result = OutputLayer.Select(n => n.OutputValue).ToArray();
        return result;
    }

    protected override void Initialize()
    {
        XavierInitializer.InitializeUniform(this);
    }

    public override void Train(double[][] trainingData, double[][] targets, int epochs, double learningRate)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double TotalLoss = 0.0;
            for (int i = 0; i < trainingData.Length; i++)
            {
                TotalLoss += BackPropagationTrainer.ApplyBackPropagation(this, trainingData[i], targets[i], learningRate, Predict);
            }

            double Mse = TotalLoss / trainingData.Length;
            Console.WriteLine($"Epoch {epoch + 1} / {epoch} - Loss(Mse): {Mse:f6}");
        }
    }
}
