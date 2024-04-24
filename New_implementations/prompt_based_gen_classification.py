import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GenerativeTextClassifier:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

    def calculate_log_probability(self, prompt, text):
        # Tokenize the concatenated prompt and text
        tokens_tensor = self.tokenizer.encode(prompt + text, return_tensors='pt')
        
        # We need to calculate the log likelihood for each token in the text, conditioned on the prompt and previous tokens
        with torch.no_grad():
            outputs = self.model(tokens_tensor[:, :-1])  # Get model output up to the last token
            logits = outputs.logits  # Model predictions before softmax
        
        # Calculate log probabilities using logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Select the logits for the actual tokens in the text
        shifted_tokens_tensor = tokens_tensor[:, 1:]  # Shifted right, excluding the first token from the prompt
        log_probs = torch.gather(log_probs, 2, shifted_tokens_tensor.unsqueeze(2)).squeeze(-1)
        
        # Sum log probabilities across tokens to get the total log likelihood
        total_log_prob = log_probs.sum().item()
        return total_log_prob

    def classify(self, text, y):
        positive_prompt = f"Write a text that contains {y}: "
        negative_prompt = f"Write a text that doesn’t contain {y}: "
        
        log_prob_positive = self.calculate_log_probability(positive_prompt, text)
        log_prob_negative = self.calculate_log_probability(negative_prompt, text)
        
        scores = torch.tensor([log_prob_positive, log_prob_negative])
        probabilities = torch.nn.functional.softmax(scores, dim=0)

        return {'contains_y': probabilities[0].item(), 'does_not_contain_y': probabilities[1].item()}

# Example usage:
classifier = GenerativeTextClassifier()
text = "This person is so stupid."
y = "insult"
probabilities = classifier.classify(text, y)
print("Probabilities:", probabilities)
