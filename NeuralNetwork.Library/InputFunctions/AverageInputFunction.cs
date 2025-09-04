using NeuralNetwork.Abstration;

namespace NeuralNetwork.Library.InputFunctions
{
    public class AverageInputFunction : IInputFunction
    {
        public double CalculateInput(IEnumerable<ISynapse> dendrites, double bias)
        {
            double weightedSum = dendrites.Sum(d => d.Value * d.Weight);
            double weightSum = dendrites.Sum(d => d.Weight);

            return (weightSum == 0 ? 0 : weightedSum / weightSum) + bias;
        }
    }
}
