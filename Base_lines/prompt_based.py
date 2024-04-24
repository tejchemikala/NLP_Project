from transformers import pipeline
import torch

class PromptBasedToxicityDetector:
    def __init__(self, model="distilbert-base-uncased-finetuned-sst-2-english"):
        # Load a model fine-tuned for sentiment analysis as a placeholder
        self.classifier = pipeline("text-classification", model=model)

    def predict_toxicity(self, text):
        # Construct the prompt
        prompt = f"Text: {text}\nQuestion: Does the above text contain toxicity?"
        
        # Predict using the classifier
        results = self.classifier(prompt)
        
        # Organize results to find probabilities of 'Yes' (toxic) and 'No' (not toxic)
        label_map = {'LABEL_0': 'No', 'LABEL_1': 'Yes'}  # Adjust these labels based on the actual model used
        probabilities = {label_map[res['label']]: res['score'] for res in results}
        yes_prob = probabilities.get('Yes', 0)
        no_prob = probabilities.get('No', 0)

        return {'Yes': yes_prob, 'No': no_prob}

# Example usage:
detector = PromptBasedToxicityDetector()
text = "This person is an idiot and a loser."
probabilities = detector.predict_toxicity(text)
print("Probabilities:", probabilities)
