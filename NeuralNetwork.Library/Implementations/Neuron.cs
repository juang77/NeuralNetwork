using NeuralNetwork.Abstration;

namespace NeuralNetwork.Library.Implementations
{
    public class Neuron(IInputFunction inputFunction, IActivationFunction activationFunction) : INeuron
    {
        public IEnumerable<ISynapse> Dendrites => DendritesField;

        private readonly List<ISynapse> DendritesField = new();

        public IAxon Axon { get; } = new Axon();

        public double OutputValue { get; private set; }

        public double Bias { get; private set; }

        public IInputFunction InputFunction => inputFunction;

        public IActivationFunction Activationfunction() => activationFunction;
       
        public void AddDendrite(ISynapse dendrite)
        {
            dendrite.OnInputValueReceived += dendrite_OnInputValueReceived;
            DendritesField.Add(dendrite);
        }

        public void AddTerminal(ISynapse terminal) => Axon.AddTerminal(terminal);

        public void SetBiasValue(double value)
        {
            Bias = value;
        }

        private void dendrite_OnInputValueReceived(ISynapse dendrite)
        {
            double inputValue = inputFunction != default ? inputFunction.CalculateInput(DendritesField, Bias) : dendrite.Value;

            OutputValue = activationFunction != default ? activationFunction.CalculateOutput(inputValue) : inputValue;

            Axon.SendOutputValueToTerminal(OutputValue);
        }
    }
}
