using NeuralNetwork.Abstration;

namespace NeuralNetwork.Library.InputFunctions
{
    public class WeightedProductInputFunction : IInputFunction
    {
        public double CalculateInput(IEnumerable<ISynapse> dendrites, double bias)
        {
            double product = 1.0;

            foreach (var d in dendrites)
                product *= Math.Pow(d.Value, d.Weight);

            return product + bias; 
        }
    }
}
