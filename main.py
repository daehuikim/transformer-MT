from testing import Training, test
import evaluate

model = Training.load_trained_model()
input_file = "prediction.txt"
output_file = "output.txt"
model_file = "Helsinki-NLP/opus-mt-en-de"
test.generate_inference(input_file,output_file,model_file)

# Read the content from prediction.txt and groundtruth.txt
with open("output.txt", "r", encoding="utf-8") as f:
    prediction = f.readlines()

with open("grth.txt", "r", encoding="utf-8") as f:
    groundtruth = f.readlines()

# Tokenize the reference and candidate translations
for i in range(len(prediction)):
    prediction[i]=prediction[i]
    if prediction[i].endswith("\n"):
        prediction[i]=prediction[i].rstrip("\n")

grth=[]
for i in range(len(groundtruth)):
    groundtruth[i]=groundtruth[i]
    if groundtruth[i].endswith("\n"):
        groundtruth[i]=groundtruth[i].rstrip("\n")
    grth.append([groundtruth[i]])

# Calculate the BLEU score
bleu = evaluate.load("bleu")
bleu_score = bleu.compute(predictions=prediction, references=grth, smooth=True)

# Print the BLEU score
print("BLEU Score:", bleu_score)