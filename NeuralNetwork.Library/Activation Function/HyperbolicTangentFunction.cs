using NeuralNetwork.Abstration;

namespace NeuralNetwork.Library.Activation_Function
{
    public class HyperbolicTangentActivationFunction : IActivationFunction
    {
        public double CalculateOutput(double input)
        {
            return Math.Tanh(input);
        }

        public double CalculateDerivative(double input)
        {
            double y = CalculateOutput(input);
            return 1 - y * y;
        }
    }
}
