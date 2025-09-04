using NeuralNetwork.Abstration;

namespace NeuralNetwork.Library.Activation_Function
{
    public class HyperbolicTangentActivationFunction : IActivationFunction
    {
        public double CalculateOutput(double input)
        {
            return Math.Tanh(input);
        }
    }
}
