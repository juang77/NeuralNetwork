namespace NeuralNetwork.Abstration
{
    public interface IInputFunction
    {
        double CalculateInput(IEnumerable<ISynapse> dendrites, double bias);
    }
}
