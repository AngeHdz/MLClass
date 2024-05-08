using Microsoft.ML;

namespace MLClass
{
    public class MLModel
    {
        public MLContext mlContext = new MLContext();

        public void Training()
        {
            Task.Run(() => {

                IDataView Dta = TellTaleModel.LoadImageFromFolder(mlContext, "C:\\Image\\Training\\");
                ITransformer d = TellTaleModel.RetrainModel(mlContext, Dta);
                mlContext.Model.Save(d, Dta.Schema, "C:\\Image\\Model\\TellTaleModel.mlnet");

                return Task.CompletedTask;
            });
        }

        public void Test(byte[] imageBytes, ref string Key,ref float Accuracy) 
        {
            //byte[] imageBytes = File.ReadAllBytes(@"C:\Ametek_Resources\Training\Auto\20240325_162334.jpg");
            TellTaleModel.ModelInput sampleData = new TellTaleModel.ModelInput()
            {
                ImageSource = imageBytes,
            };

            // Make a single prediction on the sample data and print results.
            IOrderedEnumerable<KeyValuePair<string, float>> sortedScoresWithLabel = TellTaleModel.PredictAllLabels(sampleData);

            KeyValuePair<string, float> d = sortedScoresWithLabel.OrderByDescending(x => x.Value).First();
            Key = d.Key;
            Accuracy = d.Value;
        }
    }
}
