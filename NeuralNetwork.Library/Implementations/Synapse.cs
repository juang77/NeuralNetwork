using NeuralNetwork.Abstration;

namespace NeuralNetwork.Library.Implementations
{
    internal class Synapse : ISynapse
    {
        public double Value { get; private set; }

        public double Weight { get; private set; }

        public event Action<ISynapse> OnInputValueReceived;

        public void ReceiveInputValue(double value)
        {
            Value = value;

            OnInputValueReceived?.Invoke(this);
        }

        public void SetWeightValue(double weight)
        {
            Weight = weight;
        }
    }
}
