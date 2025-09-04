using NeuralNetwork.Abstration;

namespace NeuralNetwork.Library.Activation_Function
{
    public class SoftsignActivationFunction : IActivationFunction
    {
        public double CalculateOutput(double input)
        {
            return input / (1 + Math.Abs(input));
        }
    }
}
