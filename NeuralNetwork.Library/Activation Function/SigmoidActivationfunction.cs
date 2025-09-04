using NeuralNetwork.Abstration;

namespace NeuralNetwork.Library.Activation_Function
{
    public class SigmoidActivationfunction : IActivationFunction
    {
        public double CalculateOutput(double input)
        {
            return 1.0 / (1.0 + Math.Exp(-input));
        }

        public double CalculateDerivative(double input)
        {
            double y = CalculateOutput(input);
            return y * (1 - y);
        }
    }
}
