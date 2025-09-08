using ScottPlot.Plottables;
using StudyCase.Library;
using System.Windows;

namespace WPFNeuralNetwork.App
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        Scatter[] PlotViewScatters;

        public MainWindow()
        {
            InitializeComponent();
            PlotView.Plot.Axes.SetLimits(0, 600, 0, 600);
        }

        private void btnTrain_Click(object sender, RoutedEventArgs e)
        {
            var StudyCaseExample = new StudyCase.Library.StudyCaseExample();
            StudyCaseExample.OnGetTrainingDataEvent += StudyCaseExample_OnGetTrainingDataEvent;
            StudyCaseExample.OnTrainingEvent += StudyCaseExample_OnTrainingEvent;
            StudyCaseExample.OnPredictionEvent += StudyCase_OnPredictionEvent;


            Task.Run(() => StudyCaseExample.ExecuteExample());

        }

        private void StudyCase_OnPredictionEvent(int index, double[] prediction)
        {
            Dispatcher.Invoke(() =>
            {
                var scatter = PlotViewScatters[index];
                scatter.Color = ScottPlot.Color.FromColor(prediction[0] > 0.5 ? System.Drawing.Color.Green : System.Drawing.Color.Red);
                if(index % 2000 == 0)
                    PlotView.Refresh();
            });

        }

        private void StudyCaseExample_OnTrainingEvent(StudyCase.Library.Models.OnTrainingEventArgs args)
        {
            Dispatcher.Invoke(() =>
            {
                string Text = $"Epoch: {args.TotalEpocs}, Current Epoch: {args.CurrentEpoc}, MSE: {args.Mse:f4}";
                ConsoleTextBox.AppendText(Text);
                ConsoleTextBox.AppendText(Environment.NewLine);
                ConsoleTextBox.ScrollToEnd();
            });
        }

        private void StudyCaseExample_OnGetTrainingDataEvent(double[][] trainingData)
        {
            Dispatcher.Invoke(() =>
            {
                PlotViewScatters = new Scatter[trainingData.Length];
                PlotView.Plot.Clear(); // Clear previous plots if needed
                for (int i = 0; i < trainingData.Length; i++)
                {
                    // Use the ScottPlot.Scatter extension method for Plot
                    var scatter = PlotView.Plot.Add.Scatter(trainingData[i][0] * 595, trainingData[i][1] * 595);
                    scatter.MarkerSize = 5;
                    scatter.Color = ScottPlot.Color.FromColor(System.Drawing.Color.Black);
                    PlotViewScatters[i] = scatter;
                }

                var LineV = PlotView.Plot.Add.Line(235, 415, 235, 600);
                LineV.Color = ScottPlot.Color.FromColor(System.Drawing.Color.Black);
                LineV.LineWidth = 5;

                var LineH = PlotView.Plot.Add.Line(235, 415, 600, 415);
                LineH.Color = ScottPlot.Color.FromColor(System.Drawing.Color.Black);
                LineH.LineWidth = 5;

                PlotView.Refresh();
            });
        }

        private void Window_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            double Size = PlotViewGrid.ActualHeight;
            PlotView.Width = Size;
            PlotView.Height = Size;
        }
    }
}