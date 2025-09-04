using NeuralNetwork.Abstration;

namespace NeuralNetwork.Library.Activation_Function
{
    public class LeakyReLUActivationFunction : IActivationFunction
    {
        private readonly double _alpha;
        public LeakyReLUActivationFunction(double alpha = 0.01)
        {
            _alpha = alpha;
        }
        public double CalculateOutput(double input)
        {
            return input > 0 ? input : _alpha * input;
        }

        public double CalculateDerivative(double input)
        {
            return input > 0 ? 1 : _alpha;
        }
    }
}
