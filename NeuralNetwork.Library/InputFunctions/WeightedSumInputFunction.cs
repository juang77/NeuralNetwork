using NeuralNetwork.Abstration;

namespace NeuralNetwork.Library.InputFunctions
{
    public class WeightedSumInputFunction : IInputFunction
    {
        public double CalculateInput(IEnumerable<ISynapse> dendrites, double bias) => 
            dendrites.Sum(d => d.Value * d.Weight) + bias;

    }
}
