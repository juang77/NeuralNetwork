using System.Collections.Concurrent;

namespace CSV.Library
{
    public static class CsvReader
    {
        public static IEnumerable<string[]> Read(string filePath, bool hasHeader = false)
        {
            if (!System.IO.File.Exists(filePath))
                throw new System.IO.FileNotFoundException($"Data file not found: {filePath}");

            ConcurrentBag<string[]> Data = new();
            IEnumerable<string> DataRows;

            if (hasHeader)
            {
                DataRows = File.ReadLines(filePath).Skip(1);
            }
            else
            {
                DataRows = File.ReadLines(filePath);
            }

            Parallel.ForEach(DataRows, line =>
            {
                try
                {
                    var values = line.Split(',', StringSplitOptions.TrimEntries);
                    Data.Add(values);
                }
                catch (Exception ex)
                {
                    throw new Exception($"Error reading line {line}. Error: {ex.Message}");
                }
            });

            return Data;
        }
    }
}
