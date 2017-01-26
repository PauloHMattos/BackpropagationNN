using System;
using System.Collections.Generic;
using System.IO;
using System.Xml.Serialization;

namespace NeuralNetwork.Training
{
    public struct TrainResult
    {
        public int Epochs;
        public double Error;
        public List<double> Weights;
        
        public void Serialize(string designation)
        {
            var date = $"{DateTime.Now:dd-MM-yyyy_HH-mm-ss}";
            var path = "Train Results/" + designation + "_" + date + ".xml";
            using (var stream = File.Create(path))
            {
                var serializer = new XmlSerializer(typeof(TrainResult));
                serializer.Serialize(stream, this);
            }
        }

        public static TrainResult Deserialize(string path)
        {
            TrainResult result;
            using (var stream = File.Open("Train Results/" + path + ".xml", FileMode.Open))
            {
                var serializer = new XmlSerializer(typeof(TrainResult));
                result = (TrainResult)serializer.Deserialize(stream);
            }
            return result;
        }
    }
}