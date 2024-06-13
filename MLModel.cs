using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace MLClass
{
    public class MLModel
    {
        public event EventHandler<EventMLStatus> OnReport;
        public MLContext mlContext = new MLContext();
        public string PathTraining = string.Empty;
        public string PathModel = string.Empty;
        public List<string> Categories = new List<string>();

        public MLModel() { }

        protected virtual void OnReportReached(EventMLStatus e)
        {
            if (OnReport == null)
            {
                return;
            }
            OnReport.Invoke(this, e);
        }

        

        public void Training()
        {
            var options = new ImageClassificationTrainer.Options()//Optimizacion de Parametros
            {
                LabelColumnName = "Label",
                Arch = ImageClassificationTrainer.Architecture.MobilenetV2,
                FeatureColumnName = "ImageSource",
                MetricsCallback = CallbackReport,
                ValidationSet = null,
                TestOnTrainSet = false,
                EarlyStoppingCriteria = null,
                WorkspacePath = "workspace",
                Epoch = 10,
                BatchSize = 5,
                LearningRate = 0.01f
            };
            Task.Run(() => {
                Stopwatch stopwatch = new Stopwatch(); 
                stopwatch.Start();
                IDataView Dta = TellTaleModel.LoadImageFromFolder(mlContext, PathTraining);//Procesamiento de imagenes y creacion de DataSet
                ITransformer d = TellTaleModel.RetrainModel(mlContext, Dta, options, out MulticlassClassificationMetrics metrics);//Entrenamiento con el uso de Opciones
                stopwatch.Stop();
                mlContext.Model.Save(d, Dta.Schema, PathModel);//Se guarda el modelo en un archivo para su uso
                OnReportReached(new EventMLStatus("--------------------------------------------------------------------------"));
                OnReportReached(new EventMLStatus("Finish Training Model Process"));
                OnReportReached(new EventMLStatus("--------------------------------------------------------------------------"));
                OnReportReached(new EventMLStatus($"LogLossReduction: {metrics.LogLossReduction.ToString()}"));
                OnReportReached(new EventMLStatus($"LogLoss: {metrics.LogLoss}"));
                OnReportReached(new EventMLStatus($"MicroAccuracy: {metrics.MicroAccuracy}"));
                OnReportReached(new EventMLStatus($"MacroAccuracy: {metrics.MacroAccuracy}"));
                OnReportReached(new EventMLStatus($"File Model: {PathModel}"));
                OnReportReached(new EventMLStatus(string.Format("Training elapsed: {0:hh\\:mm\\:ss\\.fff}", stopwatch.Elapsed)));
                OnReportReached(new EventMLStatus("--------------------------------------------------------------------------"));
                return Task.CompletedTask;
            });
        }

        private void CallbackReport(ImageClassificationTrainer.ImageClassificationMetrics metrics)
        {
            if (metrics.Bottleneck != null)
            {
                OnReportReached(new EventMLStatus(metrics.Bottleneck.ToString()));
            }
            if (metrics.Train != null) 
            {
                OnReportReached(new EventMLStatus(metrics.Train.ToString()));
            }
        }

        public void Test(byte[] imageBytes, ref string Key,ref float Accuracy, string PathModel = "")
        {
            Categories.Clear();
            TellTaleModel.ModelInput sampleData = new TellTaleModel.ModelInput()
            {
                ImageSource = imageBytes,
            };

            // Make a single prediction on the sample data and print results.
            IOrderedEnumerable<KeyValuePair<string, float>> sortedScoresWithLabel = TellTaleModel.PredictAllLabels(sampleData, PathModel);
            Categories = sortedScoresWithLabel.Select(x => x.Key).ToList();
            KeyValuePair<string, float> d = sortedScoresWithLabel.OrderByDescending(x => x.Value).First();
            Key = d.Key;
            Accuracy = d.Value;
        }

        public void Test(byte[] imageBytes, ref string Key, ref float Accuracy,ref List<string> log,string PathModel = "")
        {
            Categories.Clear();
            TellTaleModel.ModelInput sampleData = new TellTaleModel.ModelInput()
            {
                ImageSource = imageBytes,
            };

            // Make a single prediction on the sample data and print results.
            IOrderedEnumerable<KeyValuePair<string, float>> sortedScoresWithLabel = TellTaleModel.PredictAllLabels(sampleData, PathModel);
            Categories = sortedScoresWithLabel.Select(x => x.Key).ToList();
            foreach (var data in sortedScoresWithLabel) 
            {
                log.Add($"{data.Key}: {data.Value}");
            }

            KeyValuePair<string, float> d = sortedScoresWithLabel.OrderByDescending(x => x.Value).First();
            Key = d.Key;
            Accuracy = d.Value;
        }
    }
}
