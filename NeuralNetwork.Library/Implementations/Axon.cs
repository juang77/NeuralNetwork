using NeuralNetwork.Abstration;

namespace NeuralNetwork.Library.Implementations
{
    public class Axon : IAxon
    {
        public List<ISynapse> Terminals { get; } = [];

        public void AddTerminal(ISynapse terminal)
        {
            Terminals.Add(terminal);
        }

        public void SendOutputValueToTerminal(double value)
        {
            foreach (ISynapse terminal in Terminals)
            {
                terminal.ReceiveInputValue(value);
            }
        }
    }
}
