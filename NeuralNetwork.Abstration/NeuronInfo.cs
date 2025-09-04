namespace NeuralNetwork.Abstration
{
    public class NeuronInfo
    {
        public NeuronInfo(int layerIndex, int neuronIndex, double[] weights, double bias, double outputValue)
        {
            LayerIndex = layerIndex;
            NeuronIndex = neuronIndex;
            Weights = weights;
            Bias = bias;
            OutputValue = outputValue;
        }

        public int LayerIndex { get; }
        public int NeuronIndex { get; }
        public double[] Weights { get; }
        public double Bias { get; }
        public double OutputValue { get; }
    }
}
