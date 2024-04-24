import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np

class GenerativeToneTextClassifier:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

    def get_token_probability(self, prompt):
        """ Generate token probabilities conditioned on a given prompt. """
        tokens_tensor = self.tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(tokens_tensor[:, :-1])
            logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        shifted_tokens_tensor = tokens_tensor[:, 1:]
        selected_log_probs = torch.gather(log_probs, 2, shifted_tokens_tensor.unsqueeze(2)).squeeze(-1)
        return selected_log_probs.sum().item()

    def calculate_max_tone_probability(self, text, tones):
        """ Determine the maximum probability of a tone given the text up to each token. """
        max_log_probs = []
        for tone in tones:
            prompt = f"What is the probability of the tone being {tone}?: {text}"
            log_prob = self.get_token_probability(prompt)
            max_log_probs.append(log_prob)
        return np.max(max_log_probs)  # Get the maximum probability among tones

    def classify(self, text, y, tones):
        positive_prompt = f"Write a text that contains {y}: {text}"
        negative_prompt = f"Write a text that doesn’t contain {y}: {text}"

        log_prob_positive = self.get_token_probability(positive_prompt)
        log_prob_negative = self.get_token_probability(negative_prompt)

        max_log_prob_tone = self.calculate_max_tone_probability(text, tones)

        # Combine the log probabilities
        adjusted_log_prob_positive = log_prob_positive + max_log_prob_tone
        adjusted_log_prob_negative = log_prob_negative + max_log_prob_tone

        scores = torch.tensor([adjusted_log_prob_positive, adjusted_log_prob_negative])
        probabilities = torch.nn.functional.softmax(scores, dim=0)

        return {'contains_y': probabilities[0].item(), 'does_not_contain_y': probabilities[1].item()}

# Example usage:
classifier = GenerativeToneTextClassifier()
text = "This person is so stupid."
y = "insult"
tones = ["sarcastic", "angry", "questioning", "happy", "sad", "neutral", "excited", "anxious"]
probabilities = classifier.classify(text, y, tones)
print("Probabilities:", probabilities)
