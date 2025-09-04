using NeuralNetwork.Abstration;

namespace NeuralNetwork.Library.InputFunctions
{
    public class MinInputFunction : IInputFunction
    {
        public double CalculateInput(IEnumerable<ISynapse> dendrites, double bias)
        {
            if (!dendrites.Any())
                throw new ArgumentException("Dendrites must not be empty.");

            return dendrites.Min(d => d.Value * d.Weight) + bias;
        }
    }
}
