namespace NeuralNetwork.Abstration
{
    public interface IActivationFunction
    {
        double CalculateOutput(double input);
        double CalculateDerivative(double input);
    }
}
