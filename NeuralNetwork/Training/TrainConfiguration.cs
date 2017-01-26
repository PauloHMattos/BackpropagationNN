using System;
using System.IO;
using System.Xml.Serialization;

namespace NeuralNetwork.Training
{
    public struct TrainConfiguration
    {
        public int MaxEpochs;
        public double MinError;
        public double LearnRate;
        public double Momentum;
        public double WeightDecay;

        public static TrainConfiguration Deserialize(string path)
        {
            TrainConfiguration config;

            using (var stream = File.Open("Train Configurations/" + path + ".xml", FileMode.Open))
            {
                var serializer = new XmlSerializer(typeof(TrainConfiguration));
                config = (TrainConfiguration)serializer.Deserialize(stream);
            }

            return config;
        }

        public void Serialize(string designation)
        {
            var pattern = "Train Configurations/" + designation;

            var i = 0;
            var path = pattern;
            while (File.Exists(path + ".xml"))
            {
                path += "_";
                i++;
            }

            if(i > 0)
                path += i;

            using (var stream = File.Create(path + ".xml"))
            {
                var serializer = new XmlSerializer(typeof(TrainConfiguration));
                serializer.Serialize(stream, this);
            }
        }
    }
}