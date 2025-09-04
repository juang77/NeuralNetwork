using NeuralNetwork.Abstration;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Library.Initializers
{
    public static class XavierInitializer
    {
        public static void InitializeUniform(INeuralNetwork network)
        {
            InitializeWeightsWithXavierInitializer(network, (nin, nout) =>
            {
                double limit = Math.Sqrt(6.0) / Math.Sqrt(nin + nout);
                return (-limit, limit);
            });
            
        }

        public static void InitializeNormal(INeuralNetwork network)
        {
            InitializeWeightsWithXavierInitializer(network, (nin, nout) =>
            {
                double limit = Math.Sqrt(2.0) / Math.Sqrt(nin + nout);
                return (0, limit);
            });

        }


        private static void InitializeWeightsWithXavierInitializer(INeuralNetwork network, Func<int, int, (double min, double max)> initializer)
        {
            Random rand = new Random();
            Initialize(network.InputLayer, network.HiddenLayers[0], rand, initializer);

            for (int i = 0; i < network.HiddenLayers.Length - 1; i++)
            {
                Initialize(network.HiddenLayers[i], network.HiddenLayers[i + 1], rand, initializer);
            }

            Initialize(network.HiddenLayers.Last(), network.OutputLayer, rand, initializer);
        }

        private static void Initialize(INeuron[] inputLayer, INeuron[] neurons, Random rand, Func<int, int, (double min, double max)> initializer)
        {
            int nin = inputLayer.Length;
            int nout = neurons.Length;
            var (min, max) = initializer(nin, nout);

            foreach (var neuron in neurons)
            {
                foreach (var Terminal in neuron.Axon.Terminals)
                {
                    Terminal.SetWeightValue(rand.NextDouble() * (max - min) + min);
                }
                neuron.SetBiasValue(rand.NextDouble() * (max - min) + min);
            }
        }
    }
}
