namespace NeuralNetwork.Abstration;

public interface ISynapse
{
    void FeedInput(double value);

    void SetWeightValue(double weight);

    double value { get; }
    double weight { get; }

    event Action<ISynapse> OnFeedInput;
}
