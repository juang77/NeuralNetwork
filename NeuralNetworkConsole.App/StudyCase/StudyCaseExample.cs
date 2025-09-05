using CSV.Library;
using NeuralNetworkConsole.App.Model;

namespace NeuralNetworkConsole.App.StudyCase
{
    public static class StudyCaseExample
    {
        public static void ExecuteExample()
        {
            var Data = GetFileData();

            (var TrainingData, var TestData) = GetTrainAndTestData(Data);
            double[][] trainingInputs = ExtractInputs(TrainingData);
            double[][] trainingTargets = ExtractTargets(TrainingData);

            int epochs = 1000;
            double learningRate = 0.01;

            // Create an instance of StudyCaseNeuralNetwork before calling Train
            var neuralNetwork = new StudyCaseNeuralNetwork();
            neuralNetwork.Train(trainingInputs, trainingTargets, epochs, learningRate);

            Console.WriteLine("Training complete. Testing the network:");

            (double[] Data, int Expected)[] SampleData =
                [
                    ([0.4035,0.7300], 1),
                    ([0.4035,0.650], 0),
                    ([0.523,0.712], 1),
                    ([0.622,0.435], 0),
                    ([0.347,0.617], 0),
                    ([0.459,0.745], 1),
                    ([0.359,0.659], 0),
                    ([0.712,0.523], 0),
                    ([0.435,0.655], 0),
                    ([0.959,0.959], 1),
                    ([0.459,0.730], 1)
                ];

            foreach (var item in SampleData)
            {
                Console.Write($"[{string.Join(",", item.Data)}], Expected/Predicted:");
                Console.Write($"{item.Expected},");
                var Predicted = neuralNetwork.Predict(item.Data)[0];
                Console.WriteLine($"{(Predicted >= 0.5 ? 1 : 0)} ({Predicted})");
            }
        }

        private static double[][] ExtractTargets(IEnumerable<StudyData> trainingData)
        {
            return trainingData
                .Select(data => new double[] { (double)data.Expected })
                .ToArray();
        }

        private static double[][] ExtractInputs(IEnumerable<StudyData> trainingData)
        {
            return trainingData
                .Select(data => new double[] { data.StudyHours, data.SleepingHours })
                .ToArray();
        }

        private static (IEnumerable<StudyData> TrainingData, IEnumerable<StudyData> TestData) GetTrainAndTestData(StudyData[] data)
        {
            // Shuffle the data
            var rnd = new Random();
            var shuffled = data.OrderBy(x => rnd.Next()).ToArray();

            int trainCount = (int)(shuffled.Length * 0.7);
            var trainingData = shuffled.Take(trainCount).ToArray();
            var testData = shuffled.Skip(trainCount).ToArray();

            return (trainingData, testData);
        }

        static StudyData[] GetFileData()
        {
            var DataPath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "StudyCase\\Assets", "StudyData.csv");
            var Data = CsvReader.Read(DataPath, true);
            // Assuming CsvReader.Read returns a collection of rows, map them to StudyData[]
            return Data.Select(row => new StudyData(
                Convert.ToDouble(row[0]),
                Convert.ToDouble(row[1]),
                Convert.ToInt32(row[2])
            )).ToArray();
        }
    }
}
