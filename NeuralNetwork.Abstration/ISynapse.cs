namespace NeuralNetwork.Abstration;

public interface ISynapse
{
    void ReceiveInputValue(double value);

    void SetWeightValue(double weight);

    double Value { get; }
    double Weight { get; }

    event Action<ISynapse> OnInputValueReceived;
}
