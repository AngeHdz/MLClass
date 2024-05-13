using Microsoft.ML;

namespace MLClass
{
    public class MLModel
    {
        public MLContext mlContext = new MLContext();
        public string PathTraining = string.Empty;
        public string PathModel = string.Empty;

        public MLModel() { }
        

        public void Training()
        {
            Task.Run(() => {

                //IDataView Dta = TellTaleModel.LoadImageFromFolder(mlContext, "C:\\Image\\Training\\");
                IDataView Dta = TellTaleModel.LoadImageFromFolder(mlContext, PathTraining);
                ITransformer d = TellTaleModel.RetrainModel(mlContext, Dta);
                //mlContext.Model.Save(d, Dta.Schema, "C:\\Image\\Model\\TellTaleModel.mlnet");
                mlContext.Model.Save(d, Dta.Schema, PathModel);

                return Task.CompletedTask;
            });
        }

        public void Test(byte[] imageBytes, ref string Key,ref float Accuracy, string PathModel = "") 
        {
            TellTaleModel.ModelInput sampleData = new TellTaleModel.ModelInput()
            {
                ImageSource = imageBytes,
            };

            // Make a single prediction on the sample data and print results.
            IOrderedEnumerable<KeyValuePair<string, float>> sortedScoresWithLabel = TellTaleModel.PredictAllLabels(sampleData, PathModel);

            KeyValuePair<string, float> d = sortedScoresWithLabel.OrderByDescending(x => x.Value).First();
            Key = d.Key;
            Accuracy = d.Value;
        }

        public void Test(byte[] imageBytes, ref string Key, ref float Accuracy,ref List<string> log,string PathModel = "")
        {
            TellTaleModel.ModelInput sampleData = new TellTaleModel.ModelInput()
            {
                ImageSource = imageBytes,
            };

            // Make a single prediction on the sample data and print results.
            IOrderedEnumerable<KeyValuePair<string, float>> sortedScoresWithLabel = TellTaleModel.PredictAllLabels(sampleData, PathModel);
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
