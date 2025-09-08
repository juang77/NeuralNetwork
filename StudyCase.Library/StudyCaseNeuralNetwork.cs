using NeuralNetwork.Abstration;
using NeuralNetwork.Library.Activation_Function;
using NeuralNetwork.Library.Implementations;
using NeuralNetwork.Library.Initializers;
using NeuralNetwork.Library.Trainers;
using StudyCase.Library.Models;

namespace StudyCase.Library;

public class StudyCaseNeuralNetwork : NeuralNetworkBase, INeuralNetwork
{
    public event Action<OnTrainingEventArgs>? OnTrainingEvent;
    public event Action<int, double[]> OnPredictionEvent;

    public StudyCaseNeuralNetwork()
    {
        HiddenLayerActivationFunction = new HyperbolicTangentActivationFunction();
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
                //Forward propagation
                double[] outputs = Predict(trainingData[i]);
                OnPredictionEvent?.Invoke(i, outputs);

                //apply backpropagation
                TotalLoss += BackPropagationTrainer.ApplyBackPropagation(this, trainingData[i], targets[i], learningRate);
            }

            double Mse = TotalLoss / trainingData.Length;

            OnTrainingEvent?.Invoke(new OnTrainingEventArgs(epochs, epoch + 1, Mse));
        }
    }
}
