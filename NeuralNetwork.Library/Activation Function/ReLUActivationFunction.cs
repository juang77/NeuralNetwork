using NeuralNetwork.Abstration;

namespace NeuralNetwork.Library.Activation_Function
{
    public class ReLUActivationFunction : IActivationFunction
    {
        public double CalculateOutput(double input)
        {
            return Math.Max(0, input);
        }
    }
}
