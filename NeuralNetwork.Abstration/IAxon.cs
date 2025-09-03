namespace NeuralNetwork.Abstration;

public interface IAxon
{
    List<ISynapse> Terminals { get; }

    void AddTerminal(ISynapse terminal);

    void SendOutputValueToTerminal(double value);
}
