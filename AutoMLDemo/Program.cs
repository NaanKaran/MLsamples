using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace AutoMLDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            var counts = new[] { 0, 1, 1, 0, 2, 1, 0, 0, 10, 5, 5, 10, 1, 0, 2, 1, 0, 1 };

            var mlContext = new MLContext();
            var estimator = mlContext.Transforms.DetectIidSpike(nameof(Output.Prediction), nameof(Input.Count), confidence: 99, pvalueHistoryLength: counts.Length / 4);
            ITransformer transformer = estimator.Fit(mlContext.Data.LoadFromEnumerable(new List<Input>()));
            var input = counts.Select(x => new Input { Count = x });
            IDataView transformedData = transformer.Transform(mlContext.Data.LoadFromEnumerable(input));
            var predictions = mlContext.Data.CreateEnumerable<Output>(transformedData, false);

            foreach (var p in predictions)
            {
                Console.WriteLine($"{p.Prediction[0]}\t{p.Prediction[1]}\t{p.Prediction[2]}");
            }

            Console.WriteLine("Hello World!");
        }
    }

    class Input
    {
        public float Count { get; set; }
    }
    class Output
    {
        [VectorType(3)]
        public double[] Prediction { get; set; }
    }
}
