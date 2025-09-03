namespace NeuralNetwork.Abstration;

public interface INeuron
{
    IEnumerable<ISynapse> Dendrites { get; }

    IAxon Axon { get; }

    double OutputValue { get; }

    double Bias { get; }

    void SetBiasValue(double value);

    void AddDendrite(ISynapse dendrite);

    void AddTerminal(ISynapse terminal);

    IInputFunction InputFunction {get;}

    IActivationFunction Activationfunction();

}
