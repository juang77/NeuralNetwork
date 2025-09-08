namespace NeuralNetworkConsole.App.Model
{
    public class OnTrainingEventArgs( int totalEpocs, int currentEpoc, double mse) : EventArgs
    {
        public int TotalEpocs => totalEpocs;
        public int CurrentEpoc => currentEpoc;
        public double Mse => mse;
    }
}
