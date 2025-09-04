using NeuralNetwork.Abstration;

namespace NeuralNetwork.Library.Activation_Function
{
    public class SoftsignActivationFunction : IActivationFunction
    {
        public double CalculateOutput(double input)
        {
            return input / (1 + Math.Abs(input));
        }

        public double CalculateDerivative(double input)
        {
            double denom = 1 + Math.Abs(input);
            return 1 / (denom * denom);
        }
    }
}
