using NeuralNetwork.Abstration;

namespace NeuralNetwork.Library.Models
{
    public class ModelParameters
    {
        public int InputLayerNeuronsCount { get; set; }

        public int[] HiddenLayersNeuronsCount { get; set; }

        public int OutputLayerNeuronsCount { get; set; }

        public IEnumerable<NeuronInfo> NeuronsInfo { get; set; }
    }
}
