class ToxicTextClassifier:
    def __init__(self, lexicon_path):
        self.toxic_words = set()
        self.load_lexicon(lexicon_path)

    def load_lexicon(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as file:
                for line in file:
                    self.toxic_words.add(line.strip().lower())
        except FileNotFoundError:
            print(f"Error: The file '{path}' does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def is_toxic(self, text):
        words = text.lower().split()
        for word in words:
            if word in self.toxic_words:
                return True
        return False
    
classifier = ToxicTextClassifier('toxic_lexicon.txt')
test_text = "This is a very nice day."
if classifier.is_toxic(test_text):
    print("The text is considered toxic.")
else:
    print("The text is not considered toxic.")
